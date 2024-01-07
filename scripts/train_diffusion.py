import os
import shutil
import argparse
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch.utils.data import Subset

from datasets import get_dataset
from datasets.shape_mol_dataset import ShapeMolDataset
from datasets.shape_mol_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D
import utils.transforms as trans
import utils.misc as misc
import utils.train as utils_train
from rdkit import Chem
import time
from sklearn.metrics import roc_auc_score
import pickle
import pdb

def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='../logs_diffusion_full')
    parser.add_argument('--change_log_dir', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--continue_train_iter', type=int, default=-1)
    parser.add_argument('--train_report_iter', type=int, default=200)
    args = parser.parse_args()

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Logging
    if args.change_log_dir is not None:
        log_dir = args.change_log_dir
    else:
        log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    
    if args.change_log_dir is None:
        shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
        shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Transforms
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
    )
    train_set, val_set = subsets['train'], subsets['valid']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')
    
    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    #train_set.dataset.__getitem__(0)
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # Model
    logger.info('Building model...')
    model = ScorePosNet3D(
        config.model,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    
    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)
    
    start_iter = 1
    if args.continue_train_iter > 0 and os.path.exists(f'{ckpt_dir}/{args.continue_train_iter}.pt'):
        ckpt = torch.load(f'{ckpt_dir}/{args.continue_train_iter}.pt', map_location=args.device)
        model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
        logger.info(f'Successfully load the model! {args.continue_train_iter}.pt')
        start_iter = args.continue_train_iter + 1
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        
    print(f'ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)
            
            results = model.get_diffusion_loss(
                    ligand_pos=batch.ligand_pos,
                    ligand_v=batch.ligand_atom_feature_full,
                    batch_ligand=batch.ligand_element_batch,
                    ligand_shape=batch.shape_emb,
                    eval_mode=False
            )
            loss, loss_pos, loss_v = \
                results['loss'], results['loss_pos'], results['loss_v']
            loss = loss / config.train.n_acc_batch
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )
            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()

    def validate(it):
        sum_loss, sum_loss_pos, sum_loss_v, sum_n = 0, 0, 0, 0
        all_pred_v, all_true_v = [], []

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                t_loss, t_loss_pos, t_loss_v = [], [], []
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    
                    results = model.get_diffusion_loss(
                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        ligand_shape=batch.shape_emb,
                        eval_mode=True,
                        time_step=time_step
                    )
                    loss, loss_pos, loss_v = \
                        results['loss'], results['loss_pos'], results['loss_v']
                    
                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_n += batch_size
                    
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | '
            'Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, atom_auroc
            )
        )
        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v', avg_loss_v, it)
        writer.flush()
        return avg_loss

    try:
        best_loss, best_iter = None, None
        for it in range(start_iter, config.train.max_iters + 1):
            # with torch.autograd.detect_anomaly():
            train(it)

            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')
    except KeyboardInterrupt:
        logger.info('Terminating...')
