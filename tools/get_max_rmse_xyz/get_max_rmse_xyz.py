"""
    Purpose:
        Get the points with the largest errors in the force_train.out,
        virial_train.out or energy_train.out file, and find the ID of
        the training set to which these points belong and output it.
        Now, added function to selectively output remaining structures.
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


def Print_MAX_xyz(nf_mark, num_lines, train_xyz, fout="find_out.xyz", fres=None):

    fout_str = ""
    if fres != None:
        fout_not = ""
    with open(train_xyz, "r") as fi:
        flines = fi.readlines()
        for fi in range(len(nf_mark)):
            flsta = num_lines[fi]
            if fi == len(num_lines):
                flend = len(flines)
            else:
                try:
                    flend = num_lines[fi+1]
                except:
                    flend = len(train_xyz)
            if nf_mark[fi]:
                fout_str += "".join(flines[flsta:flend])
            else:
                if fres != None:
                    fout_not += "".join(flines[flsta:flend])

    fo = open(fout, 'w')
    fo.write(fout_str)
    fo.close()
    if fres != None:
        fr = open(fres, 'w')
        fr.write(fout_not)
        fr.close()


fxyz = sys.argv[1]
floss = sys.argv[2]
nmax = int(sys.argv[3])

rmse_max, rmse_ids, file_type = Get_rmse_ids(nmax, floss)
num_lines, num_atoms = Get_fram_line(fxyz)
sum_atoms = np.cumsum(num_atoms)

nf_mark = np.zeros((len(num_lines)))
for i in rmse_ids:
    if file_type == "force":
        nfram = np.searchsorted(sum_atoms, i)
    else:
        nfram = i
    nf_mark[nfram] = 1

#print(f"The lagerest RMSE with {nmax} atom are located in fram {sel_flist}")
Print_MAX_xyz(nf_mark, num_lines, fxyz, fout='find_out.xyz', fres='reserve.xyz')
