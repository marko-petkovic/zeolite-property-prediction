import os
import sys
sys.path.append(os.getcwd())
sys.path.insert(0, "../zeolite-property-prediction/code/")
sys.path.insert(0, "../zeolite-property-prediction/")

import numpy as np

from time import time

import torch



from models.equivariant_mpnn import MPNN, MPNNPORE

from utils.ZeoliteData import get_zeolite, get_data_pore, get_data_graph, get_data_megnet
from utils.dataloading import get_data, get_graph_data

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-z', '--zeolite', choices=['MOR', 'MFI', 'RHO','ITW' ], type=str, default='MOR')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-r', '--repetitions', type=int, default=1)

    args = parser.parse_args()


    print('loading data...')
    data = get_zeolite(args.zeolite, sym=True)

    ref = data['ref'] # reflections
    tra = data['tra'] # translations
    l = data['l'] # scale of the unit cell
    ang = data['ang'] if 'ang' in data else None

    print('creating graph data...')
    atoms, hoa, X, A, d, X_pore, A_pore, d_pore, pore = get_data(l, args.zeolite, ang)
    edges, idx1, idx2, idx2_oh = get_graph_data(A, d)
    edges_sp, idx1_sp, idx2_sp, idx2_oh_sp = get_graph_data(A_pore, d_pore)
    edges_ps, idx1_ps, idx2_ps, idx2_oh_ps = get_graph_data(A_pore.T, d_pore.T)

    print('loading model...')
    mpnn = MPNNPORE(idx1, idx2, idx2_oh, X, X_pore, ref, tra,
                            idx1_sp, idx2_sp, idx2_oh_sp, 
                            idx1_ps, idx2_ps, idx2_oh_ps,
                            hid_size=[8]*6, site_emb_size=8, edge_emb_size=8, mlp_size=24,
                            centers=10, mx_d=6, width=1, pool='sum', pool_pore=True, site_pred=True)
    _, _, trainloader = get_data_pore(atoms, hoa, edges, pore, edges_sp, edges_ps, bs=args.batch_size, p=1.0, random=True)


    sites, bonds, sites_p, bonds_sp, bonds_ps, y = next(iter(trainloader))  
    
    for i in range(args.repetitions):
        start = time()
        _ = mpnn(sites.float(), bonds.float(), sites_p.float(), bonds_sp.float(), bonds_ps.float())
        end = time()
        print('Time for one forward pass:', end-start)
