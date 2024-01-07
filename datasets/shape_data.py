import os
import pickle
import pandas as pd
from typing import Any
import lmdb
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.shape import *
from multiprocessing import Pool
from functools import partial
import trimesh
from sklearn.neighbors import KDTree

class ShapeDataset(Dataset):

    def __init__(self, config):
        super().__init__()
        self.raw_path = config.path
        self.processed_path = os.path.dirname(self.raw_path) + '/' + config.data_name + f'_processed_{config.version}.lmdb'
        self.db = None
        self.size = 0
        self.keys = None
        self.config = config

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

    def _connect_db(self):
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

    def __len__(self):
        return len(self.data)
    
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=self.config.datasize*(1024*1024*1024),
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        
        num_skipped = 0
        start_idx = 0
        atom_stamp = get_atom_stamp(grid_resolution=self.config.grid_resolution, max_dist=4)
        all_mols = pickle.load(open(self.raw_path, 'rb'))['rdkit_mol_cistrans_stereo']
        batch_size = 10000

        with db.begin(write=True) as txn:
            for i in tqdm(range(0, len(all_mols), batch_size)):
                for j, mol in enumerate(all_mols[i:min(len(all_mols), i+batch_size)]):
                    try:
                        data_dict = {'smiles': Chem.MolToSmiles(mol)}
                        if self.config.shape_type == 'voxel':
                            data_dict['voxel'] = get_voxel_shape(mol, atom_stamp, 
                                                grid_resolution=self.config.grid_resolution, 
                                                max_dist=self.config.max_dist)
                        elif self.config.shape_type == 'point_cloud' or self.config.shape_type == 'mesh':
                            data_dict['mesh'] = get_mesh(mol)
                        
                        tensor_shape_dict = torchify_dict(data_dict)
                                    
                        txn.put(
                            key=str(i+j).encode(),
                            value=pickle.dumps(tensor_shape_dict)
                        )
                    except Exception as e:
                        print(e)
                        num_skipped += 1
                        print('Skipping (%d) %s: %d' % (num_skipped, tensor_shape_dict['smiles'], start_idx + i))
                        continue
                txn.commit()
                try:
                    txn = db.begin(write=True)
                except:
                    break
        db.close()

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data['id'] = idx
        
        return data

class ShapeData:
    def __init__(self, batch_data, config):
        self._store = {'points':[], 'values': [], config.shape_type: []}
        
        for i, data in enumerate(batch_data):
            point_cloud = None
            for k, v in data.items():
                if k == 'mesh' and config.shape_type == 'point_cloud':
                    point_cloud = get_pointcloud_from_mesh(data['mesh'], config.point_cloud_samples).squeeze(0)
                    continue
                if k != config.shape_type: continue

                if i == 0: self._store[k] = []
                self._store[k].append(v)

            if config.shape_type == 'voxel':
                points, values = sample_grids_for_voxel(data['voxel'], config.num_samples)
            elif config.shape_type == 'point_cloud':
                points, values = sample_points_for_pointcloud(data['mesh'], point_cloud, config.num_samples, config.loss_type)

            if config.shape_type == 'point_cloud':
                offset = torch.mean(point_cloud, dim=0)
                point_cloud = point_cloud - offset
                points = points - offset

                self._store[config.shape_type].append(point_cloud)

            self._store['points'].append(points)
            self._store['values'].append(values)

        for k, v in self._store.items():
            if not isinstance(v[0], torch.Tensor): continue
            self._store[k] = torch.stack(v, dim=0)

    def to(self, device):
        for k, v in self._store.items():
            if isinstance(v, torch.DoubleTensor):
                v = v.float()
            if isinstance(v, torch.Tensor):
                self._store[k] = v.to(device)

    def __getattr__(self, key):
        return self._store[key]
    
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, mapping):
        for key, value in mapping.items():
            self.__dict__[key] = value
            

def sample_points_for_pointcloud(mesh, point_clouds, num_samples, loss_type):
    """
    sample query points from molecule surface
    """
    mesh = trimesh.Trimesh(mesh[0], mesh[1])
    
    # half of the samples are within the mesh
    withinmesh_points = []
    outmesh_points = []
    points = np.random.random((num_samples*3, 3)) * (mesh.extents + 2) + mesh.bounds[0] - 1
    contained = mesh.contains(points)
    
    withinmesh_points = points[contained]
    outmesh_points = points[~contained]
    withinmesh_samples = min(int(num_samples / 2), len(withinmesh_points))
    outmesh_samples = num_samples - withinmesh_samples

    points = np.concatenate((withinmesh_points[:withinmesh_samples, :], outmesh_points[:outmesh_samples, :]), axis=0)
    points = torch.from_numpy(points)
    
    # get objectives
    if loss_type == 'occupancy':
        values = np.concatenate((np.ones(withinmesh_samples), np.zeros(outmesh_samples)), axis=0)
        values = torch.from_numpy(values)
    elif loss_type == 'signed_distance':
        kdtree = KDTree(point_clouds)
        distances, _ = kdtree.query(points)
        sign = np.concatenate((np.ones(withinmesh_samples), -1 * np.ones(outmesh_samples)), axis=0)
        values = torch.from_numpy(sign * distances.squeeze(-1))
    return points, values

def sample_grids_for_voxel(voxel, num_samples):
    sampled_random_points = torch.randint(0, 45, (int(num_samples / 2), 3))
    
    sampled_random_values = voxel[sampled_random_points[:, 0], sampled_random_points[:, 1], sampled_random_points[:, 2]]
    
    shape_points = torch.stack(torch.where(voxel > 0), axis=1)
    sampled_shape_point_idxs = torch.randint(0, shape_points.shape[0], (int(num_samples / 2),))
    sampled_shape_points = torch.index_select(shape_points, 0, sampled_shape_point_idxs)
    sampled_shape_values = torch.LongTensor([1] * sampled_shape_point_idxs.shape[0])

    points = torch.concat((sampled_random_points, sampled_shape_points), axis=0)
    values = torch.concat((sampled_random_values, sampled_shape_values), axis=0)
    
    return points, values

def collate_fn(batch_data, config):
    shape_batch_data = ShapeData(batch_data, config)
    return shape_batch_data

def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output