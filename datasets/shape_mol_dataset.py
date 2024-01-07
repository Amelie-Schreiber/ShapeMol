import os
import pdb
import oddt
import pickle
import lmdb
import time
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import copy
from utils.data import *
from .shape_mol_data import ShapeMolData, torchify_dict
from utils.shape import *
from functools import partial
from multiprocessing import Pool
from utils.subproc_shapeAE import SubprocShapeAE

class ShapeMolDataset(Dataset):

    def __init__(self, config, transform):
        super().__init__()
        self.config = config
        self.raw_path = config.path.rstrip('/')
        self.processed_dir = config.processed_path
        self.processed_path = os.path.join(self.processed_dir,
                                           os.path.basename(self.raw_path).split(".")[-2] + f'_processed_{config.version}.lmdb')
        self.transform = transform
        self.db = None

        self.keys = None
        
        self.shape_type = config.shape.shape_type
        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=self.config.datasize*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
            self.size = len(self.keys)
        

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
    
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=self.config.datasize*(1024*1024*1024),   # 20GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )

        self._process_mose(db)
    
    def _process_mose(self, db):
        shape_func, subproc_voxelae = get_shape_func(self.config.shape)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            if 'test' in self.raw_path:
                all_mols = pickle.load(open(self.raw_path, 'rb'))
            else:
                all_mols = pickle.load(open(self.raw_path, 'rb'))['rdkit_mol_cistrans_stereo']
            
            batch = self.config.shape.batch_size * self.config.shape.num_workers
            chunk_size = self.config.chunk_size
            
            for chunk_id, i in enumerate(range(0, len(all_mols), chunk_size)):
                print(f'processing chunk {chunk_id}....')
                chunk_mols = all_mols[i:min(len(all_mols), i+chunk_size)]

                pool = Pool(processes=self.config.num_workers)
                chunk_dicts = []
                for data in tqdm(pool.imap(parse_rdkit_mol, chunk_mols)):
                    chunk_dicts.append(data)
                pool.close()
                print("finish rdkit parse")
                for j in tqdm(range(i, min(len(all_mols), i+chunk_size), batch)):
                    batch_mols = all_mols[j:min(j+batch, len(all_mols))]
                    batch_dicts = chunk_dicts[j-i:min(j+batch, len(all_mols))-i]
                    
                    if len(batch_mols) == 0: continue
                    batch_shape_embs, batch_bounds, batch_pointclouds, batch_pointcloud_centers = shape_func(batch_mols)
                    
                    for k, ligand_dict in enumerate(batch_dicts):
                        #try:
                        data = ShapeMolData.from_ligand_dicts(
                            ligand_dict=torchify_dict(ligand_dict),
                        )
                        data.shape_emb = batch_shape_embs[k]
                        data.ligand_pos = data.ligand_pos - batch_pointcloud_centers[k]
                        
                        data.bound = batch_bounds[k]

                        if 'test' in self.raw_path:
                            data.point_cloud = batch_pointclouds[k]
                            data.mol = batch_mols[k]

                        data.ligand_index = torch.tensor(i+j+k)
                        data = data.to_dict()  # avoid torch_geometric version issue
                        
                        txn.put(
                            key=str(i+j+k).encode(),
                            value=pickle.dumps(data)
                        )
            
            if self.config.shape.shape_parallel: subproc_voxelae.close()
        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ShapeMolData(**data)
        data.id = idx
        
        shape_emb = data.shape_emb
        if self.transform is not None:
            data = self.transform(data)
                
        
        data.shape_emb = shape_emb
        return data
        

def get_shape_func(config):
    if config.shape_type == 'electroshape':
        shape_func = get_electro_shape_emb
    elif config.shape_type == 'voxelAE_shape':
        atom_stamp = get_atom_stamp(grid_resolution=config.grid_resolution, max_dist=4)
        if config.shape_parallel: shapeae = SubprocShapeAE(config)
        else: shapeae = build_voxel_shapeAE_model(config, device='cuda')
        
        shape_func = partial(get_voxelAE_shape_emb,
                             model=shapeae,
                             atom_stamp=atom_stamp,
                             grid_resolution=config.grid_resolution,
                             max_dist=config.max_dist,
                             batch_size=config.batch_size,
                             shape_parallel=config.shape_parallel
                             )
    elif config.shape_type == 'pointAE_shape':
        if config.shape_parallel: shapeae = SubprocShapeAE(config)
        else: shapeae = build_point_shapeAE_model(config, device='cuda')
        shape_func = partial(get_pointAE_shape_emb,
                             model=shapeae,
                             point_cloud_samples=config.point_cloud_samples,
                             config=config,
                             batch_size=config.batch_size,
                             shape_parallel=config.shape_parallel
                             )
    return shape_func, shapeae
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    dataset = ShapeMolDataset(args.path)
    print(len(dataset), dataset[0])
