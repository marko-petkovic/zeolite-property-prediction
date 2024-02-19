import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

import os
import sys
sys.path.append(os.getcwd())
sys.path.insert(0, "../zeolite-property-prediction/code/")
sys.path.insert(0, "../zeolite-property-prediction/")


from utils.ZeoliteData import get_zeolite
from utils.dataloading import get_data


S1 = 50
S2 = 80

L1 = 2.5
L2 = 1.75

def plot(zeo, ax):

    data = get_zeolite(zeo, True)
    
    ref = data['ref'] # reflections
    tra = data['tra'] # translations
    l = data['l'] # scale of the unit cell
    ang = data['ang'] if 'ang' in data else None # angles of unit cell
    
    # specific for MOR
    atoms, hoa, X, A, d, X_pore, A_pore, d_pore, pore = get_data(l, zeo, ang)

    if zeo == 'MOR':
        plot_mor(X,A,X_pore,A_pore,l,ref,tra,ax)

    elif zeo == 'MFI':
        plot_mfi(X,A,X_pore,A_pore,l,ref,tra,ax)
    elif zeo == 'RHO':
        plot_rho(X,A,X_pore,A_pore,l,ref,tra,ax)
    elif zeo == 'ITW':
        plot_itw(X,A,X_pore,A_pore,l,ref,tra,ax)
    
    # edges, idx1, idx2, idx2_oh = get_graph_data(A, d)
    # edges_sp, idx1_sp, idx2_sp, idx2_oh_sp = get_graph_data(A_pore, d_pore)
    # edges_ps, idx1_ps, idx2_ps, idx2_oh_ps = get_graph_data(A_pore.T, d_pore.T)


def plot_mor(X, A, X_pore, A_pore, l, ref, tra, ax):

    # colors for atoms
    color = np.zeros((X.shape[0],))
    for i in range(len(X)):
        
        for j in range(4):
            
            for k in range(ref.shape[0]):
                
                if np.abs(X[i] - (np.mod(ref[k]*X[j] + tra[k], 1))).sum() < 0.001:
                    
                    color[i] = j
                    
                    break
    # colors for pores
    color2 = np.zeros((X_pore.shape[0],))
    for i in range(len(X_pore)):
        
        for j in [0,2,4,15]:
            
            for k in range(ref.shape[0]):
                
                if np.abs(X_pore[i] - (np.mod(ref[k]*X_pore[j] + tra[k], 1))).sum() < 0.001:
                    
                    color2[i] = j
                    
                    break

    # atom-atom edge colors
    ce_dict = {}
    row, col = np.nonzero(A)
    n_c = 0
    for i in range(len(row)):
        
        added = False
        
        r,c = row[i], col[i]
        
        for j in ce_dict:
            
            if added: break
            
            test = ce_dict[j][0]
            
            test_r = X[test[0]]
            test_c = X[test[1]]
            
            
            for k in range(ref.shape[0]):
                
                r1 = np.mod(ref[k]*X[r] + tra[k], 1)
                r2 = np.mod(ref[k]*X[c] + tra[k], 1)
                
                if np.abs(r1-test_r).sum() < 0.001 and np.abs(r2-test_c).sum() < 0.001:
                    
                    ce_dict[j].append((r,c))
                    
                    added = True
                    break
            
        if not added:
            ce_dict[n_c] = [(r,c)]
            n_c += 1

    # atom-pore edge colors
    cp_dict = {}
    row, col = np.nonzero(A_pore)
    n_c = 0
    for i in range(len(row)):
        
        added = False
        
        r,c = row[i], col[i]
        
        for j in cp_dict:
            
            if added: break
            
            test = cp_dict[j][0]
            
            test_r = X[test[0]]
            test_c = X_pore[test[1]]
            
            
            for k in range(ref.shape[0]):
                
                r1 = np.mod(ref[k]*X[r] + tra[k], 1)
                r2 = np.mod(ref[k]*X_pore[c] + tra[k], 1)
                
                if np.abs(r1-test_r).sum() < 0.001 and np.abs(r2-test_c).sum() < 0.001:
                    
                    cp_dict[j].append((r,c))
                    
                    added = True
                    break
            
        if not added:
            cp_dict[n_c] = [(r,c)]
            n_c += 1 


    #plt.figure(figsize=(18.256/4, 20.534/4))
    for i in ce_dict:
        x = 0
        for j in ce_dict[i]:
            
            x1, x2 = X[j[0]].copy(), X[j[1]].copy()
            
            d = x2-x1
            
            for _d in range(3):
                
                if d[_d] > 0.5:
                    
                    x2[_d] -= 1
                
                elif d[_d] < -.5:
                    
                    x2[_d] += 1
                
            
            
            
            if x == 0:
                
                line = ax.plot([x1[0], x2[0]], [x1[1], x2[1]], lw=L1, zorder=5)
                ax.plot([x1[0], x2[0]], [x1[1], x2[1]], lw=L1*1.5, c='black', zorder=4)
            
            ax.plot([x1[0], x2[0]], [x1[1], x2[1]], c= line[0].get_c(),lw=L1, zorder=5)
            ax.plot([x1[0], x2[0]], [x1[1], x2[1]], c= 'black',lw=L1*1.5, zorder=4)
            x+=1
            
            
    for i in cp_dict:
        x = 0
        for j in cp_dict[i]:
            
            x1, x2 = X[j[0]].copy(), X_pore[j[1]].copy()
            
            d = x2-x1
            
            for _d in range(3):
                
                if d[_d] > 0.5:
                    
                    x2[_d] -= 1
                
                elif d[_d] < -.5:
                    
                    x2[_d] += 1
                
            
            
            
            if x == 0:
                line = ax.plot([x1[0], x2[0]], [x1[1], x2[1]],lw=L2,ls='--', path_effects=[pe.Stroke(linewidth=L2*1.5, foreground='black'), pe.Normal()])
            
            ax.plot([x1[0], x2[0]], [x1[1], x2[1]], c= line[0].get_c(),lw=L2,ls='--', path_effects=[pe.Stroke(linewidth=L2*1.5, foreground='black'), pe.Normal()])
            #     line = plt.plot([x1[0], x2[0]], [x1[1], x2[1]],lw=2,ls='--', path_effects=[pe.Stroke(linewidth=4, foreground='black')])
            
            # plt.plot([x1[0], x2[0]], [x1[1], x2[1]], c= line[0].get_c(),lw=2,ls='--', path_effects=[pe.Stroke(linewidth=4, foreground='black')])
            x+=1
            
    
    ax.scatter(X_pore[:,0], X_pore[:,1], s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
    ax.scatter(X_pore[:,0]+1, X_pore[:,1], s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
    ax.scatter(X_pore[:,0], X_pore[:,1]+1, s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
    ax.scatter(X_pore[:,0]+1, X_pore[:,1]+1, s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    ax.scatter(X[:,0], X[:,1], c=color, s=S2, zorder=10, edgecolors='black')
    


def plot_mfi(X, A, X_pore, A_pore, l, ref, tra, ax):
    _X = X[X[:,1]>0.75]
    _A = A[X[:,1]>0.75]
    _A = _A[:,X[:,1]>0.75]
    _A_pore = A_pore[X[:,1]>0.75]
    # atom color
    color = np.zeros((_X.shape[0],))
    for i in range(len(_X)):
    
        yy = False
        for j in range(12):
            
            for k in range(ref.shape[0]):
                
                if np.abs(_X[i] - (np.mod(ref[k]*X[j] + tra[k], 1))).sum() < 0.01:
                # if (X[i] == (np.mod(ref[k]*X[j] + tra[k], 1))).all():
                    
                    color[i] = j
                    yy = True  
                    break
    
        if yy is not True:
            color[i] = -1
    
    # pore color
    color2 = np.zeros((X_pore.shape[0]-2,))
    for i in range(len(X_pore)-2):
        
        for j in range(len(X_pore)-2):
            
            for k in range(ref.shape[0]):
    
                if np.abs(X_pore[i] - (np.mod(ref[k]*X_pore[j] + tra[k], 1))).sum() < 0.001:
                    
                    color2[i] = j
                    
                    break

    # atom atom edge color
    ce_dict = {}
    row, col = np.nonzero(_A)
    n_c = 0
    for i in range(len(row)):
        
        added = False
        
        r,c = row[i], col[i]
        
        for j in ce_dict:
            
            if added: break
            
            test = ce_dict[j][0]
            
            test_r = _X[test[0]]
            test_c = _X[test[1]]
            
            
            for k in range(ref.shape[0]):
                
                r1 = np.mod(ref[k]*_X[r] + tra[k], 1)
                r2 = np.mod(ref[k]*_X[c] + tra[k], 1)
                
                if np.abs(r1-test_r).sum() < 0.001 and np.abs(r2-test_c).sum() < 0.001:
                    
                    ce_dict[j].append((r,c))
                    
                    added = True
                    break
            
        if not added:
            ce_dict[n_c] = [(r,c)]
            n_c += 1     
    
    # atom pore color
    cp_dict = {}
    row, col = np.nonzero(_A_pore[:,:12])
    n_c = 0
    for i in range(len(row)):
        
        added = False
        
        r,c = row[i], col[i]
        
        for j in cp_dict:
            
            if added: break
            
            test = cp_dict[j][0]
            
            test_r = _X[test[0]]
            test_c = X_pore[test[1]]
            
            
            for k in range(ref.shape[0]):
                
                r1 = np.mod(ref[k]*_X[r] + tra[k], 1)
                r2 = np.mod(ref[k]*X_pore[c] + tra[k], 1)
                
                if np.abs(r1-test_r).sum() < 0.001 and np.abs(r2-test_c).sum() < 0.001:
                    
                    cp_dict[j].append((r,c))
                    
                    added = True
                    break
            
        if not added:
            cp_dict[n_c] = [(r,c)]
            n_c += 1
    
    # plotting
    for i in ce_dict:
        x = 0
        for j in ce_dict[i]:
            
            x1, x2 = _X[j[0]].copy(), _X[j[1]].copy()
            
            d = x2-x1
            
            for _d in range(3):
                
                if d[_d] > 0.5:
                    
                    x2[_d] -= 1
                
                elif d[_d] < -.5:
                    
                    x2[_d] += 1
                
            
            
            
            if x == 0:
                
                line = ax.plot([x1[2], x2[2]], [x1[0], x2[0]], lw=L1, zorder=5)
                ax.plot([x1[2], x2[2]], [x1[0], x2[0]], lw=L1*1.5, c='black', zorder=4)
    
                ax.plot([x1[2]-1, x2[2]-1], [x1[0], x2[0]], c= line[0].get_c(), lw=L1, zorder=5)
                plt.plot([x1[2]-1, x2[2]-1], [x1[0], x2[0]], lw=L1*1.5, c='black', zorder=4)
    
                ax.plot([x1[2]+1, x2[2]+1], [x1[0], x2[0]], c= line[0].get_c(), lw=L1, zorder=5)
                ax.plot([x1[2]+1, x2[2]+1], [x1[0], x2[0]], lw=L1*1.5, c='black', zorder=4)
            
            ax.plot([x1[2], x2[2]], [x1[0], x2[0]], c= line[0].get_c(),lw=L1, zorder=5)
            ax.plot([x1[2], x2[2]], [x1[0], x2[0]], c= 'black',lw=L1*1.5, zorder=4)
            ax.plot([x1[2]-1, x2[2]-1], [x1[0], x2[0]], c= line[0].get_c(),lw=L1, zorder=5)
            ax.plot([x1[2]-1, x2[2]-1], [x1[0], x2[0]], c= 'black',lw=L1*1.5, zorder=4)
            ax.plot([x1[2]+1, x2[2]+1], [x1[0], x2[0]], c= line[0].get_c(),lw=L1, zorder=5)
            ax.plot([x1[2]+1, x2[2]+1], [x1[0], x2[0]], c= 'black',lw=L1*1.5, zorder=4)
            x+=1
            
            
    for i in cp_dict:
        x = 0
        for j in cp_dict[i]:
            
            x1, x2 = _X[j[0]].copy(), X_pore[j[1]].copy()
            
            d = x2-x1
            
            for _d in range(3):
                
                if d[_d] > 0.5:
                    
                    x2[_d] -= 1
                
                elif d[_d] < -.5:
                    
                    x2[_d] += 1
                
            
            
            
            if x == 0:
                line = ax.plot([x1[2], x2[2]], [x1[0], x2[0]],lw=L2,ls='--', path_effects=[pe.Stroke(linewidth=L2*1.5, foreground='black'), pe.Normal()])
                ax.plot([x1[2]-1, x2[2]-1], [x1[0], x2[0]],lw=L2,ls='--', c= line[0].get_c(), path_effects=[pe.Stroke(linewidth=L2*1.5, foreground='black'), pe.Normal()])
                ax.plot([x1[2]+1, x2[2]+1], [x1[0], x2[0]],lw=L2,ls='--', c= line[0].get_c(), path_effects=[pe.Stroke(linewidth=L2*1.5, foreground='black'), pe.Normal()])
            
            ax.plot([x1[2], x2[2]], [x1[0], x2[0]], c= line[0].get_c(),lw=L2,ls='--', path_effects=[pe.Stroke(linewidth=L2*1.5, foreground='black'), pe.Normal()])
            ax.plot([x1[2]-1, x2[2]-1], [x1[0], x2[0]], c= line[0].get_c(),lw=L2,ls='--', path_effects=[pe.Stroke(linewidth=L2*1.5, foreground='black'), pe.Normal()])
            ax.plot([x1[2]+1, x2[2]+1], [x1[0], x2[0]], c= line[0].get_c(),lw=L2,ls='--', path_effects=[pe.Stroke(linewidth=L2*1.5, foreground='black'), pe.Normal()])
            x+=1
            
    
    ax.scatter(X_pore[:12,2], X_pore[:12,0], s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
    ax.scatter(X_pore[:12,2]+1, X_pore[:12,0], s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
    ax.scatter(X_pore[:12,2], X_pore[:12,0]+1, s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
    ax.scatter(X_pore[:12,2]+1, X_pore[:12,0]+1, s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    ax.scatter(_X[:,2], _X[:,0], c=color, s=S2, zorder=10, edgecolors='black')



def plot_rho(X, A, X_pore, A_pore, l, ref, tra, ax):
    color = np.zeros((X.shape[0],))
    for i in range(len(X)):
        
        for j in range(4):
            
            for k in range(ref.shape[0]):
                
                if np.abs(X[i] - (np.mod(X[j]@ref[k] + tra[k], 1))).sum() < 0.001:
                    
                    color[i] = j
                    
                    break
    
    
    color2 = np.zeros((X_pore.shape[0],))
    for i in range(len(X_pore)):
        
        for j in [0,1]:
            
            for k in range(ref.shape[0]):
                
                if np.abs(X_pore[i] - (np.mod(X_pore[j]@ref[k] + tra[k], 1))).sum() < 0.001:
                    
                    color2[i] = j
                    
                    break


    ce_dict = {}
    row, col = np.nonzero(A)
    n_c = 0
    for i in range(len(row)):
        
        added = False
        
        r,c = row[i], col[i]
        
        for j in ce_dict:
            
            if added: break
            
            test = ce_dict[j][0]
            
            test_r = X[test[0]]
            test_c = X[test[1]]
            
            
            for k in range(ref.shape[0]):
                
                r1 = np.mod(X[r]@ref[k] + tra[k], 1)
                r2 = np.mod(X[c]@ref[k] + tra[k], 1)
                
                if np.abs(r1-test_r).sum() < 0.05 and np.abs(r2-test_c).sum() < 0.05:
                    
                    ce_dict[j].append((r,c))
                    
                    added = True
                    break
            
        if not added:
            ce_dict[n_c] = [(r,c)]
            n_c += 1  
    
    cp_dict = {}
    row, col = np.nonzero(A_pore)
    n_c = 0
    for i in range(len(row)):
        
        added = False
        
        r,c = row[i], col[i]
        
        for j in cp_dict:
            
            if added: break
            
            test = cp_dict[j][0]
            
            test_r = X[test[0]]
            test_c = X_pore[test[1]]
            
            
            for k in range(ref.shape[0]):
                
                r1 = np.mod(X[r]@ref[k] + tra[k], 1)
                r2 = np.mod(X_pore[c]@ref[k] + tra[k], 1)
                
                if np.abs(r1-test_r).sum() < 0.001 and np.abs(r2-test_c).sum() < 0.001:
                    
                    cp_dict[j].append((r,c))
                    
                    added = True
                    break
            
        if not added:
            cp_dict[n_c] = [(r,c)]
            n_c += 1


#plt.figure(figsize=(18.256/4, 20.534/4))
    for i in ce_dict:
        x = 0
        for j in ce_dict[i]:
            
            x1, x2 = X[j[0]].copy(), X[j[1]].copy()
            
            d = x2-x1
            
            for _d in range(3):
                
                if d[_d] > 0.5:
                    
                    x2[_d] -= 1
                
                elif d[_d] < -.5:
                    
                    x2[_d] += 1
                
            
            
            
            if x == 0:
                
                line = ax.plot([x1[0], x2[0]], [x1[1], x2[1]], lw=L1, zorder=5)
                ax.plot([x1[0], x2[0]], [x1[1], x2[1]], lw=L1*1.5, c='black', zorder=4)
                #cs.append(line[0].get_c())
            ax.plot([x1[0], x2[0]], [x1[1], x2[1]], c= line[0].get_c(),lw=L1, zorder=5)
            ax.plot([x1[0], x2[0]], [x1[1], x2[1]], c= 'black',lw=L1*1.5, zorder=4)
            x+=1
            
            
    for i in cp_dict:
        x = 0
        for j in cp_dict[i]:
            
            x1, x2 = X[j[0]].copy(), X_pore[j[1]].copy()
            
            d = x2-x1
            
            for _d in range(3):
                
                if d[_d] > 0.5:
                    
                    x2[_d] -= 1
                
                elif d[_d] < -.5:
                    
                    x2[_d] += 1
                
            
            
            
            if x == 0:
                line = ax.plot([x1[0], x2[0]], [x1[1], x2[1]],lw=L2,ls='--', path_effects=[pe.Stroke(linewidth=L2*1.5, foreground='black'), pe.Normal()])
                
            ax.plot([x1[0], x2[0]], [x1[1], x2[1]], c= line[0].get_c(),lw=L2,ls='--', path_effects=[pe.Stroke(linewidth=L2*1.5, foreground='black'), pe.Normal()])
            x+=1
            
    
    ax.scatter(X_pore[:,0], X_pore[:,1], s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
    ax.scatter(X_pore[:,0]+1, X_pore[:,1], s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
    ax.scatter(X_pore[:,0], X_pore[:,1]+1, s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
    ax.scatter(X_pore[:,0]+1, X_pore[:,1]+1, s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    ax.scatter(X[:,0], X[:,1], c=color, s=S2, zorder=10, edgecolors='black')


def plot_itw(X, A, X_pore, A_pore, l, ref, tra, ax):

    _X_pore = X_pore[:2]
    _A_pore = A_pore[:,:2]

    color = np.zeros((X.shape[0],))
    for i in range(len(X)):
        
        for j in range(3):
            
            for k in range(ref.shape[0]):
                
                if np.abs(X[i] - (np.mod(X[j]*ref[k] + tra[k], 1))).sum() < 0.1:
                    
                    color[i] = j
                    
                    break
    
    
    color2 = np.zeros((_X_pore.shape[0],))
    for i in range(len(_X_pore)):
        
        for j in range(5):
            
            for k in range(ref.shape[0]):
                
                if np.abs(_X_pore[i] - (np.mod(X_pore[j]*ref[k] + tra[k], 1))).sum() < 0.001:
                    
                    color2[i] = j
                    
                    break

    ce_dict = {}
    row, col = np.nonzero(A)
    n_c = 0
    for i in range(len(row)):
        
        added = False
        
        r,c = row[i], col[i]
        
        for j in ce_dict:
            
            if added: break
            
            test = ce_dict[j][0]
            
            test_r = X[test[0]]
            test_c = X[test[1]]
            
            
            for k in range(ref.shape[0]):
                
                r1 = np.mod(X[r]*ref[k] + tra[k], 1)
                r2 = np.mod(X[c]*ref[k] + tra[k], 1)
                
                if np.abs(r1-test_r).sum() < 0.05 and np.abs(r2-test_c).sum() < 0.05:
                    
                    ce_dict[j].append((r,c))
                    
                    added = True
                    break
            
        if not added:
            ce_dict[n_c] = [(r,c)]
            n_c += 1
    
    cp_dict = {}
    row, col = np.nonzero(_A_pore)
    n_c = 0
    for i in range(len(row)):
        
        added = False
        
        r,c = row[i], col[i]
        
        for j in cp_dict:
            
            if added: break
            
            test = cp_dict[j][0]
            
            test_r = X[test[0]]
            test_c = _X_pore[test[1]]
            
            
            for k in range(ref.shape[0]):
                
                r1 = np.mod(ref[k]*X[r] + tra[k], 1)
                r2 = np.mod(ref[k]*_X_pore[c] + tra[k], 1)
                
                if np.abs(r1-test_r).sum() < 0.001 and np.abs(r2-test_c).sum() < 0.001:
                    
                    cp_dict[j].append((r,c))
                    
                    added = True
                    break
            
        if not added:
            cp_dict[n_c] = [(r,c)]
            n_c += 1

    
    for i in ce_dict:
        x = 0
        for j in ce_dict[i]:
            
            x1, x2 = X[j[0]].copy(), X[j[1]].copy()
            
            d = x2-x1
            
            for _d in range(3):
                
                if d[_d] > 0.5:
                    
                    x2[_d] -= 1
                
                elif d[_d] < -.5:
                    
                    x2[_d] += 1
                
            
            
            
            if x == 0:
                
                line = ax.plot([x1[0], x2[0]], [x1[1], x2[1]], lw=L1, zorder=5)
                plt.plot([x1[0], x2[0]], [x1[1], x2[1]], lw=L1*1.5, c='black', zorder=4)
            
            ax.plot([x1[0], x2[0]], [x1[1], x2[1]], c= line[0].get_c(),lw=L1, zorder=5)
            ax.plot([x1[0], x2[0]], [x1[1], x2[1]], c= 'black',lw=L1*1.5, zorder=4)
            x+=1
            
            
    for i in cp_dict:
        x = 0
        for j in cp_dict[i]:
            
            x1, x2 = X[j[0]].copy(), X_pore[j[1]].copy()
            
            d = x2-x1
            
            for _d in range(3):
                
                if d[_d] > 0.5:
                    
                    x2[_d] -= 1
                
                elif d[_d] < -.5:
                    
                    x2[_d] += 1
                
            
            
            
            if x == 0:
                line = ax.plot([x1[0], x2[0]], [x1[1], x2[1]],lw=L2,ls='--', path_effects=[pe.Stroke(linewidth=L2*1.5, foreground='black'), pe.Normal()])
            
            ax.plot([x1[0], x2[0]], [x1[1], x2[1]], c= line[0].get_c(),lw=L2,ls='--', path_effects=[pe.Stroke(linewidth=L2*1.5, foreground='black'), pe.Normal()])
            x+=1
            
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    ax.scatter(X[:,0], X[:,1], c=color, s=S2, zorder=10, edgecolors='black')
    
    ax.scatter(X_pore[:2,0], X_pore[:2,1], s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
    ax.scatter(X_pore[:2,0]+1, X_pore[:2,1], s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
    ax.scatter(X_pore[:2,0], X_pore[:2,1]+1, s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
    ax.scatter(X_pore[:2,0]+1, X_pore[:2,1]+1, s=S1, zorder=10, c=color2, marker='s', edgecolors='black')
