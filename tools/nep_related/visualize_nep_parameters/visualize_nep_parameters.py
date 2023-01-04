# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams.update({"font.size":20})  
    
nep = np.genfromtxt("nep.restart")[:,0]

def visualize_nep_parameters(nep, N_type,basis_size_radial,basis_size_angular,n_max_radial,n_max_angular,neuron,l_max):
    
    # calculated numbers
    dim_radial = n_max_radial + 1
    l_max_reduced = l_max[0] +sum([1 if i!=0 else 0 for i in l_max[1:]])
    dim_angular = (n_max_angular + 1) * l_max_reduced
    dim = dim_radial + dim_angular
    N_ann = (dim + 2) * neuron*N_type + 1
    # neural network parameters
    para_ann=nep[0:N_ann]
    for n in range(1,N_type+1):
        offset = (n-1)*(dim + 2) * neuron
        para_w0=para_ann[offset:offset+neuron*dim].reshape((neuron,dim))
        plt.figure(figsize=(8, 6))
        plt.title('absolute connection weight value - type %d'%(n),fontsize=20)
        plt.xlabel('neuron index',fontsize=20)
        plt.ylabel('absolute connection weight value',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        for i in range(dim):
            plt.plot(np.arange(1,neuron+1),abs(para_w0[:,i]),marker=".",markersize= 20)
        plt.show()
    # descriptor parameters
    para_c=nep[N_ann:] #reshape is different from matlab
    para_c=para_c.reshape((int(para_c.shape[0]/(N_type*N_type)),N_type*N_type)).T
    para_c_radial=para_c[:,:(n_max_radial+1)*(basis_size_radial+1)]
    para_c_angular=para_c[:,(n_max_radial+1)*(basis_size_radial+1):]        
    
    # radial part with g_n in x
    plt.figure(figsize=(8, 6))
    for n in range(1,N_type*N_type+1):
        plt.subplot(N_type,N_type,n)
        plt.xlabel('g$_{n}$ index',fontsize=10)
        plt.ylabel('absolute c$_{nk}$ value',fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylim(0, abs(para_c_radial).max())
        for i in range(dim):
            plt.plot(np.arange(1,n_max_radial+2),abs(para_c_radial[n-1,:].reshape((n_max_radial+1,basis_size_radial+1))),marker=".",markersize= 10,clip_on=False)
    plt.show()

    # radial part with f_k in x
    plt.figure(figsize=(8, 6))
    for n in range(1,N_type*N_type+1):
        plt.subplot(N_type,N_type,n)
        plt.xlabel('f$_{k}$ index',fontsize=10)
        plt.ylabel('absolute c$_{nk}$ value',fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylim(0, abs(para_c_radial).max())
        for i in range(dim):
            plt.plot(np.arange(1,basis_size_radial+2),abs(para_c_radial[n-1,:].reshape((n_max_radial+1,basis_size_radial+1)).T),marker=".",markersize= 10,clip_on=False)
    plt.show()

    # angular part with g_n in x
    plt.figure(figsize=(8, 6))
    for n in range(1,N_type*N_type+1):
        plt.subplot(N_type,N_type,n)
        plt.xlabel('g$_{n}$ index',fontsize=10)
        plt.ylabel('absolute c$_{nk}$ value',fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylim(0, abs(para_c_angular).max())
        for i in range(dim):
            plt.plot(np.arange(1,n_max_radial+2),abs(para_c_angular[n-1,:].reshape((n_max_radial+1,basis_size_radial+1))),marker=".",markersize= 10,clip_on=False)
    plt.show()

    # angular part with f_k in x
    plt.figure(figsize=(8, 6))
    for n in range(1,N_type*N_type+1):
        plt.subplot(N_type,N_type,n)
        plt.xlabel('f$_{k}$ index',fontsize=10)
        plt.ylabel('absolute c$_{nk}$ value',fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylim(0, abs(para_c_angular).max())
        for i in range(dim):
            plt.plot(np.arange(1,basis_size_radial+2),abs(para_c_angular[n-1,:].reshape((n_max_radial+1,basis_size_radial+1)).T),marker=".",markersize= 10,clip_on=False)
    plt.show()

if __name__ == "__main__":
    parms_from_nep_in = """
    type       	2 Pb Te	
    n_max       4 4 
    basis_size  8 8 
    l_max       4 2 0 
    neuron      10    
    """
    sep = "\r\n" if "\r\n" in parms_from_nep_in else "\n"
    raw_parms_dict = { i.split()[0]:i.split()[1:] for i in parms_from_nep_in.split(sep) if i.strip()!=''}
    parms_dict = {}
    default_parms_dict = {'n_max':(4,4),'basis_size':(8,8),'l_max':(4,2,0),'neuron':30} 
    if 'type' not in raw_parms_dict.keys():
        raise SystemError("Exit! Type should be specified.")
    else:   
        parms_dict['type'] = int(raw_parms_dict['type'][0])
    for i in ['n_max','basis_size','l_max','neuron']:
        if i not in raw_parms_dict.keys():
            parms_dict[i] =  default_parms_dict[i]
        else:
            parms_dict[i] =  tuple(map(int,raw_parms_dict[i]))
    if parms_dict['l_max'][0]!= 4: 
        raise SystemError("Exit! L_max for three-body must be 4.")
    visualize_nep_parameters(nep,N_type=parms_dict['type'],basis_size_radial=parms_dict['basis_size'][0],basis_size_angular=parms_dict['basis_size'][1],n_max_radial=parms_dict['n_max'][0],n_max_angular=parms_dict['n_max'][1],neuron=parms_dict['neuron'][0],l_max=parms_dict['l_max'])

