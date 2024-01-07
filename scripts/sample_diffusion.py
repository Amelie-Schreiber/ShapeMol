import pdb
import os, sys
import argparse
import math
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import shutil
from glob import glob
import time
import pickle
import utils.misc as misc
import utils.transforms as trans
from utils.shape import get_voxel_shape, get_pointcloud_from_mesh, get_pointcloud_from_mol, get_mesh, get_atom_stamp, build_point_shapeAE_model
from utils.reconstruct import reconstruct_from_generated
from datasets import get_dataset
from functools import partial
from torch_geometric.transforms import Compose
from torch_geometric.data import Batch
from torch_scatter import scatter_sum, scatter_mean
from datasets.shape_mol_data import FOLLOW_BATCH
import trimesh
from sklearn.neighbors import KDTree
import numpy as np

from models.molopt_score_model import ScorePosNet3D, log_sample_categorical

def get_voxel_size(mol):
    atom_stamp = get_atom_stamp(0.5, 4)
    voxel_size = np.sum(get_voxel_shape(mol, atom_stamp, 0.5, 11))
    return voxel_size

def sample_atom_nums(batch_size, atom_nums, atom_dist):
    return np.random.choice(atom_nums, batch_size, p=atom_dist).tolist()
    

def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='none', sample_func=None,
                            threshold_type=None, threshold_args=None, sample_num_atoms='prior', use_grad=False,
                            grad_step=1000, grad_lr=1, shape_AE=None, use_mesh_data=None, use_mesh_gap=None, use_pointcloud_data=None,
                            init_scale=False, guide_stren=0):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_pos_cond_traj, all_pred_v_cond_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    
    current_i = data.ligand_element.shape[0]
    n_range = list(range(current_i // 2, int(1.5 * current_i)))
    collate_exclude_keys = ['mol', 'ligand_index', 'id']
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], exclude_keys=collate_exclude_keys, follow_batch=list(FOLLOW_BATCH) + ['bound']).to(device)
        batch_bounds = batch.bound.view(len(batch.ligand_smiles), -1, 2)
        t1 = time.time()
        #with torch.no_grad():
        if sample_num_atoms == 'size':
            assert sample_func is not None
            ligand_num_atoms = sample_func(n_data)
            batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
        elif sample_num_atoms == 'ref':
            batch_ligand = batch.ligand_element_batch
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
        else:
            raise ValueError
        
        # init ligand pos
        all_ligand_atoms = sum(ligand_num_atoms)
        
        init_ligand_pos = torch.randn(all_ligand_atoms, 3).to(device)
        
        # init ligand v
        if pos_only:
            # init_ligand_v = F.one_hot(batch.ligand_atom_feature_full, num_classes=model.num_classes).float()
            init_ligand_v = batch.ligand_atom_feature_full
        else:
            if model.v_mode == 'gaussian':
                init_ligand_v = torch.randn(len(batch_ligand), model.num_classes).to(device)
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)
        
        r = model.sample_diffusion(
            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            batch_ligand=batch_ligand,
            ligand_shape=batch.shape_emb,
            threshold_type=threshold_type,
            threshold_args=threshold_args,
            num_steps=num_steps,
            center_pos_mode=center_pos_mode,
            use_grad=use_grad,
            grad_step=grad_step,
            grad_lr=grad_lr,
            shape_AE=shape_AE,
            use_mesh_data=use_mesh_data,
            use_pointcloud_data=use_pointcloud_data,
            guide_stren=guide_stren,
            bounds=batch_bounds,
        )
        ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
        ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
        ligand_pos_cond_traj = r['pos_cond_traj']
        ligand_v_cond_traj = r['v_cond_traj']

        # unbatch pos
        ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
        ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
        try:
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]] for k in range(n_data)]  # num_samples * [num_atoms_i, 3]
        except:
            pdb.set_trace()

        all_step_pos = [[] for _ in range(n_data)]
        all_step_cond_pos = [[] for _ in range(n_data)]
        #all_step_uncond_pos = [[] for _ in range(n_data)]

        for pos, cond_pos in zip(ligand_pos_traj, ligand_pos_cond_traj):  # step_i
            p_array = pos.detach().cpu().numpy().astype(np.float64)
            cond_p_array = cond_pos.detach().cpu().numpy().astype(np.float64)
            for k in range(n_data):
                all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]])
                all_step_cond_pos[k].append(cond_p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]])

        all_step_pos = [np.stack(step_pos) for step_pos in all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
        all_step_cond_pos = [np.stack(step_pos) for step_pos in all_step_cond_pos]  # num_samples * [num_steps, num_atoms_i, 3]
        
        all_pred_pos_traj += [p for p in all_step_pos]
        all_pred_pos_cond_traj += [p for p in all_step_cond_pos]

        # unbatch v
        ligand_v_array = ligand_v.cpu().numpy()
        all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]] for k in range(n_data)]

        all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
        all_pred_v_traj += [v for v in all_step_v]

        all_step_cond_v = unbatch_v_traj(ligand_v_cond_traj, n_data, ligand_cum_atoms)
        all_pred_v_cond_traj += [v for v in all_step_cond_v]

        if not pos_only:
            all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
            all_pred_v0_traj += [v for v in all_step_v0]
            all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
            all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list, \
        all_pred_pos_cond_traj, all_pred_v_cond_traj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-i', '--data_id', type=int)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs_test')
    args = parser.parse_args()
    
    tmp_path = os.path.join(args.result_path, f'result_{args.data_id}.pt')
    if os.path.exists(tmp_path): sys.exit(0)

    logger = misc.get_logger('evaluate')
    
    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    if 'train_config' in config.model:
        logger.info(f"Load training config from: {config.model['train_config']}")
        ckpt['config'] = misc.load_config(config.model['train_config'])
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    if 'transform' in ckpt['config'].data:
        ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    else:
        ligand_atom_mode = 'full'

    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # Load dataset
    test_set = get_dataset(
        config=config.data,
        transform=transform
    )
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')
    
    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    logger.info(f'Successfully load the model! {config.model.checkpoint}')
    
    dists = pickle.load(open("./data/MOSES2_training_val_shape_atomnum_dict.pkl", 'rb'))
    
    if config.sample.use_grad:
        shapeae = build_point_shapeAE_model(config.data.shape, device='cuda', detach=False)
    else:
        shapeae = None
    
    data = test_set[args.data_id]
    data['point_cloud'] = data['point_cloud'].cpu()
    if config.sample.use_mesh:
        mesh = get_mesh(data['mol'], probe_radius=0.5)
        point_clouds = np.array(data['point_cloud'].squeeze(0))
        kdtree = KDTree(point_clouds)
        mesh = trimesh.Trimesh(mesh[0], mesh[1])
        use_mesh_data = (mesh, point_clouds, kdtree)
        config.sample.use_pointcloud = False
    else:
        use_mesh_data = None

    if config.sample.use_pointcloud:
        atom_pos = np.array(data['ligand_pos'])
        point_clouds = get_pointcloud_from_mol(atom_pos)
        kdtree = KDTree(point_clouds)
        use_pointcloud_data = (point_clouds, kdtree, config.sample.use_pointcloud_radius)
    else:
        use_pointcloud_data = None

    voxel_shape = get_voxel_size(data['mol'])
    atom_nums = {}
    for key in dists.keys():
        if key < voxel_shape + 200 and key > voxel_shape - 200:
            atom_nums.update(dists[key])
    atom_num_keys = list(atom_nums.keys())
    total_num = sum([atom_nums[key] for key in atom_num_keys])
    sample_atom_dist = [atom_nums[num]/total_num for num in atom_num_keys]
    sample_func = partial(sample_atom_nums, atom_nums=atom_num_keys, atom_dist = sample_atom_dist)
    
    if not config.sample.use_grad:
        config.sample.grad_lr = 0

    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, \
         pred_pos_cond_traj, pred_v_cond_traj = sample_diffusion_ligand(
        model, data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        threshold_type=config.sample.threshold_CFG.type if 'threshold_CFG' in config.sample else None,
        threshold_args=config.sample.threshold_CFG if 'threshold_CFG' in config.sample else None,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        use_grad=config.sample.use_grad,
        grad_step=config.sample.grad_step,
        grad_lr=config.sample.grad_lr,
        shape_AE=shapeae,
        use_mesh_data=use_mesh_data,
        use_pointcloud_data=use_pointcloud_data,
        use_mesh_gap=config.sample.use_mesh_gap,
        sample_num_atoms=config.sample.sample_num_atoms,
        sample_func=sample_func,
        guide_stren=config.sample.guide_stren
    )
    print('time: ', time_list)
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'time': time_list,
        'pred_ligand_pos_cond_traj': pred_pos_cond_traj,
        'pred_ligand_v_cond_traj': pred_v_cond_traj,
        # 'pred_ligand_v0_traj': pred_v0_traj,
        # 'pred_ligand_vt_traj': pred_vt_traj
    }
    logger.info('Sample done!')

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    #shutil.copyfile(glob(os.path.join(os.path.dirname(os.path.dirname(config.model.checkpoint)), '*.yml'))[0],
    #                os.path.join(result_path, 'training.yml'))

    torch.save(result, os.path.join(result_path, f'result_{args.data_id}.pt'))
