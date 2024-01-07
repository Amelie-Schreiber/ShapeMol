import torch
import openbabel.openbabel as ob


from .transforms import get_atomic_number_from_index

def connect_covalent_graph(ligand_pos, ligand_v, atom_mode='add_aromatic', gamma=0.2):
    atomic_index = torch.where(ligand_v > 0)[1]
    atomic_nums = get_atomic_number_from_index(atomic_index, atom_mode)
    covalent_radius = torch.FloatTensor([ob.GetCovalentRad(atomic_num) for atomic_num in atomic_nums]).unsqueeze(0).to(ligand_pos)
    
    pair_dists = torch.cdist(ligand_pos, ligand_pos, p=2)

    covalent_dists = covalent_radius + covalent_radius.transpose(1, 0) + gamma
    
    edge_mask = (pair_dists < covalent_dists) & (~torch.eye(len(atomic_nums)).to(ligand_pos).bool())
    
    edges = torch.vstack(torch.where(edge_mask))
    return edges
