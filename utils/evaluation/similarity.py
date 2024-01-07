import numpy as np
from rdkit import Chem, DataStructs
from .shaep_utils import *
from utils.espsim import GetEspSim, GetShapeSim
from rdkit.Chem import rdMolDescriptors
import pdb

def tanimoto_sim(mol, ref):
    fp1 = Chem.RDKFingerprint(ref)
    fp2 = Chem.RDKFingerprint(mol)
    return DataStructs.TanimotoSimilarity(fp1,fp2)
    

def tanimoto_sim_N_to_1(mols, ref):
    sim = [tanimoto_sim(m, ref) for m in mols]
    return sim

def tanimoto_sim_pairwise(mols):
    sims = np.ones((len(mols), len(mols)))

    for i, m1 in enumerate(mols):
        for j, m2 in enumerate(mols[i+1:]):
            sims[i, i+j+1] = tanimoto_sim(m1, m2)
            sims[i+j+1, i] = sims[i, i+j+1]
    return sims

def batched_number_of_rings(mols):
    n = []
    for m in mols:
        n.append(Chem.rdMolDescriptors.CalcNumRings(m))
    return np.array(n)

def calculate_shaep_shape_sim(mols, ref):
    aligned_mols = []
    aligned_simROCS = []
    for i, mol in enumerate(mols):
        try:
            mol, rocs = ESP_shape_align(ref, mol)
        except Exception as e:
            print(e)
            mol = None
            rocs = -1
        aligned_mols.append(mol)
        aligned_simROCS.append(rocs)
    return aligned_mols, aligned_simROCS

def calculate_espsim_shape_sim(mols, ref):
    #ref_crippen = rdMolDescriptors._CalcCrippenContribs(ref)
    aligned_simEsps = []
    aligned_simShapes = []
    for i, mol in enumerate(mols):
        #mol_crippen = rdMolDescriptors._CalcCrippenContribs(mol)
        if mol is None:
            simEsp, simShape = -1, -1
        else:
            simEsp = GetEspSim(ref, mol, prbCid = 0, refCid = 0, partialCharges = 'ml', nocheck=True)
            simShape = GetShapeSim(ref, mol, prbCid = 0, refCid = 0)
        
        aligned_simEsps.append(simEsp)
        aligned_simShapes.append(simShape)
    return aligned_simEsps, aligned_simShapes