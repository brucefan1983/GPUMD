"""
    Purpose:
        Get the points with the largest errors in the force_train.out,
        virial_train.out or energy_train.out file, and find the ID of
        the training set to which these points belong and output it.
    Notice:
        the number of output trajectories is less than or equal
        to the number of error points
    Run:
        python get_max_rmse_xyz.py train.xyz force_train.out 13
        python get_max_rmse_xyz.py train.xyz virial_train.out 13 
        python get_max_rmse_xyz.py train.xyz energy_train.out 13 
"""

import numpy as np
import sys


def Get_rmse_ids(nmax, file_force_loss):

    frmse = np.loadtxt(file_force_loss)
    if frmse.shape[1] == 6:
        rmse = np.sum(np.abs(frmse[:,0:2]-frmse[:,3:5]), axis=1)
        file_type = "force"
    elif frmse.shape[1] == 12:
        rmse = np.sum(np.abs(frmse[:,0:5]-frmse[:,6:11]), axis=1)
        file_type = "virial"
    elif frmse.shape[1] == 2:
        rmse = np.abs(frmse[:,0]-frmse[:,1])
        file_type = "energy"
    else:
        raise "Error in outfiles."
    rmse_max_ids = np.argsort(-rmse)

    return rmse[rmse_max_ids[:nmax]], rmse_max_ids[:nmax], file_type


def Get_fram_line(train_xyz):

    num_lines, num_atoms = [], []
    with open(train_xyz, "r") as fi:
        flines = fi.readlines()
        for i, line in enumerate(flines):
            if "energy" in line or "Energy" in line:
                num_lines.append(i-1)
                num_atoms.append(int(flines[i-1]))

    return num_lines, num_atoms


def Print_MAX_xyz(fram_list, num_lines, train_xyz, fout="find_out.xyz"):

    fout_str = ""
    with open(train_xyz, "r") as fi, open(fout, 'w') as fo:
        flines = fi.readlines()
        for fi in fram_list:
            flsta = num_lines[fi]
            if fi == len(num_lines):
                flend = len(flines)
            else:
                try:
                    flend = num_lines[fi+1]
                except:
                    flend = len(train_xyz)
            fout_str += "".join(flines[flsta:flend])

        fo.write(fout_str)


fxyz = sys.argv[1]
floss = sys.argv[2]
nmax = int(sys.argv[3])

rmse_max, rmse_ids, file_type = Get_rmse_ids(nmax, floss)
num_lines, num_atoms = Get_fram_line(fxyz)
sum_atoms = np.cumsum(num_atoms)

fram_list = []
for i in rmse_ids:
    if file_type == "force":
        nfram = np.searchsorted(sum_atoms, i)
    else:
        nfram = i
    if nfram not in fram_list:
        fram_list.append(nfram)

#print(f"The lagerest RMSE with {nmax} atom are located in fram {fram_list}")
Print_MAX_xyz(fram_list, num_lines, fxyz, fout='find_out.xyz')
