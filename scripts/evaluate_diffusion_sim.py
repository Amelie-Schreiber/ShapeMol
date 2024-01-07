import argparse
import os
import pdb
import pickle
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter
import warnings
from utils.evaluation import eval_bond_length
from utils.evaluation import eval_atom_type
from utils.evaluation import analyze
from utils import misc
from utils.evaluation import scoring_func
from utils import reconstruct
from utils import transforms
from utils.evaluation import similarity
from multiprocessing import Pool
import time

def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')

def get_ref_similarity(eval_tuple):
    mols, ref = eval_tuple
    results = []
    shaep_align_mols, shaep_shape_simROCS = similarity.calculate_shaep_shape_sim(mols, ref)

    # calculate pairwise similarities
    pairwise_sims = similarity.tanimoto_sim_pairwise(mols)
    
    for mol, shaep_align_mol, shaep_shape_simrocs in \
        zip(mols, shaep_align_mols, shaep_shape_simROCS):
        try:
            smiles = Chem.MolToSmiles(shaep_align_mol)
            tanimoto_sim = similarity.tanimoto_sim(mol, ref)
        except:
            tanimoto_sim = -1
            smiles = None

        try:
            chem_results = scoring_func.get_chem(mol)
        except:
            chem_results = None

        results.append({
            'smiles': smiles,
            'align_mol': shaep_align_mol,
            'tanimoto_sim': tanimoto_sim,
            'shaep_rocssim': shaep_shape_simrocs,
            'chem_results': chem_results,
        })
    return results, pairwise_sims

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('testset_path', type=str)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--basic_mode', type=eval, default=True)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--docking', type=eval, default=False)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'eval_sim_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')
    
    # Load generated data
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)
    
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    test_data = pickle.load(open(args.testset_path, 'rb'))
    test_idx_file = open("../index_map.txt", 'r')
    test_idx_dict = {}
    for line in test_idx_file.readlines():
        idxs = line.strip().split(":")
        test_idx_dict[int(idxs[0])] = int(idxs[1])
    test_idx_file.close()

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    all_results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    complete_mol_2dsims, complete_mol_3dsims = [], []
    all_smiles = []
    
    all_evalsim_tuples = []
    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        try:
            r = torch.load(r_name)  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
        except Exception as e:
            print(f'failed to load {r_name} due to error: {e}')
            continue
        cond_mol = test_data[test_idx_dict[example_idx]]

        all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
        # all_pred_ligand_pos = r['pred_ligand_pos']
        all_pred_ligand_v = r['pred_ligand_v_traj']
        num_samples += len(all_pred_ligand_pos)
        
        results, complete_mols, complete_smiles = [], [], []
        
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]

            # stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)

            all_atom_types += Counter(pred_atom_type)

            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist

            # reconstruction
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic,
                                                             basic_mode=args.basic_mode)
                smiles = Chem.MolToSmiles(mol)
            except reconstruct.MolReconsError:
                if args.verbose:
                    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                continue
            n_recon_success += 1

            if '.' in smiles:
                continue
            n_complete += 1
            complete_mols.append(mol)
            all_smiles.append(smiles)

            # now we only consider complete molecules as success
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist

            success_pair_dist += pair_dist
            success_atom_types += Counter(pred_atom_type)

            results.append({
                'mol': mol,
                'smiles': smiles,
                'ligand_filename': r_name,
                'pred_pos': pred_pos,
                'pred_v': pred_v,
            })

        all_results.append(results)
        all_evalsim_tuples.append((complete_mols, cond_mol))
    
    # now we calculate the 3d shape similarity between complete molecules and condition molecules
    complete_mol_2dsims = []
    with Pool(processes=args.num_workers) as pool:
        for i, (results, pairwise_sims) in tqdm(enumerate(pool.imap(get_ref_similarity, all_evalsim_tuples))):
            complete_mol_2dsims.append(pairwise_sims)
            for j in range(len(all_results[i])):
                all_results[i][j].update(results[j])

    logger.info(f'Evaluate done! {num_samples} samples in total.')
    
    fraction_mol_stable = all_mol_stable / num_samples
    fraction_atm_stable = all_atom_stable / all_n_atom
    fraction_recon = n_recon_success / num_samples
    fraction_eval = n_eval_success / num_samples
    fraction_complete = n_complete / num_samples
    fraction_uniq = len(set(all_smiles)) / n_complete
    avg_pairwise_sims = np.mean([(np.sum(sims)-sims.shape[0]) / (sims.shape[0] * (sims.shape[0] - 1)) for sims in complete_mol_2dsims])
    avg_ref_tanimoto_sims = np.mean([np.mean([element['tanimoto_sim'] for element in results if element['tanimoto_sim'] >= 0]) for results in all_results])
    ref_shaep_sims = [np.mean([element['shaep_rocssim'] for element in results if element['shaep_rocssim'] >= 0]) for results in all_results]
    avg_ref_shaep_sims = np.mean(ref_shaep_sims)
    std_ref_shaep_sims = np.std(ref_shaep_sims)
    avg_ref_max_shaep_sims = np.mean([np.max([element['shaep_rocssim'] for element in results if element['shaep_rocssim'] >= 0]) for results in all_results])
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
        'complete': fraction_complete,
        'uniq_over_complete': fraction_uniq,
        'avg_pairwise_sims': avg_pairwise_sims,
        'avg_ref_tanimoto_sims': avg_ref_tanimoto_sims,
        'avg_ref_shaep_rocssims': avg_ref_shaep_sims,
        'std_ref_shaep_rocssims': std_ref_shaep_sims,
        'avg_ref_max_shaep_rocssims': avg_ref_max_shaep_sims,
    }
    print_dict(validity_dict, logger)

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    logger.info('JS bond distances of complete mols: ')
    print_dict(c_bond_length_dict, logger)

    pair_length_profile = eval_bond_length.get_pair_length_profile(all_pair_dist)
    js_metrics = eval_bond_length.eval_pair_length_profile(pair_length_profile)
    logger.info('JS pair distances:  ')
    print_dict(js_metrics, logger)

    atom_type_js = eval_atom_type.eval_atom_type_distribution(all_atom_types)
    logger.info('Atom type JS: %.4f' % atom_type_js)

    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    logger.info('\nSuccess JS pair distances:  ')
    print_dict(success_js_metrics, logger)

    atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
    logger.info('Atom type JS: %.4f' % atom_type_js)

    if args.save:
        eval_bond_length.plot_distance_hist(pair_length_profile,
                                            metrics=js_metrics,
                                            save_path=os.path.join(result_path, f'pair_dist_hist_{args.eval_step}.png'))

    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        n_recon_success, n_complete, len(all_results)))
    
    mean_qed = np.mean([r['chem_results']['qed'] for results in all_results for r in results])
    mean_sa = np.mean([r['chem_results']['sa'] for results in all_results for r in results])
    logger.info('QED: %.3f SA: %.3f' % (mean_qed, mean_sa))

    # check ring distribution
    print_ring_ratio([r['chem_results']['ring_size'] for results in all_results for r in results], logger)

    if args.save:
        torch.save({
            'stability': validity_dict,
            'bond_length': all_bond_dist,
            'atom_type': all_atom_types,
            'all_results': all_results,
            'pairwise_sims': complete_mol_2dsims,
        }, os.path.join(result_path, f'metrics_{args.eval_step}_{args.basic_mode}.pt'))