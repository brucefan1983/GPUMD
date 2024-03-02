#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: LiangTing
2021/12/18 12:06:31
"""
from pylab import *
from ase.io import read
from thermo.gpumd.preproc import add_basis, repeat
from thermo.gpumd.io import create_basis, create_kpoints, ase_atoms_to_gpumd

def read_structure(data_file):

    VioletP_unitcell = read(data_file)
    VioletP_unitcell.center()
    VioletP_unitcell.wrap()
    VioletP_unitcell.pbc = [True, True, True]
    VioletP_unitcell.write("lammps-data", format='lammps-data')    # For eigenvector view

    add_basis(VioletP_unitcell)
    VioletP = repeat(VioletP_unitcell, [1, 3, 1])

    ase_atoms_to_gpumd(VioletP, M=500, cutoff=10)   # output xyz.in
    print("**************** xyz.in generated DONE! ****************")

    # For phonon dispersion
    create_basis(VioletP)
    print("**************** basis.in generated DONE! ****************")

    linear_path, sym_points, labels = create_kpoints(VioletP_unitcell, path='GY', npoints=101)
    print("**************** kpoints.in generated DONE! ****************")

    return linear_path, sym_points, labels

aw = 2
fs = 16
font = {'size': fs}
matplotlib.rc('font', **font)
matplotlib.rc('axes', linewidth=aw)

def set_fig_properties(ax_list):
    tl = 8
    tw = 2
    tlm = 4

    for ax in ax_list:
        ax.tick_params(which='major', length=tl, width=tw)
        ax.tick_params(which='minor', length=tlm, width=tw)
        ax.tick_params(which='both', axis='both', direction='in', right=True, top=True)

def plot_dispersion_gpumd(linear_path, sym_points):

    # load data
    data = np.loadtxt("omega2.out")

    for i in range(len(data)):
        for j in range(len(data[0])):

            data[i, j] = np.sqrt(abs(data[i, j])) / (2 * np.pi) * np.sign(data[i, j])

    nu = data

    figure(figsize=(8, 10))
    set_fig_properties([gca()])
    # vlines(sym_points, ymin=-0.2, ymax=60, linestyle="--", colors="pink")
    # print(nu[0, 4])
    plot(linear_path, nu[:, 0], color='C0', lw=2, label="Tersoff-1989")
    plot(linear_path, nu[:, 1:], color='C0', lw=2)
    xlim([0, max(linear_path)])
    gca().set_xticks(sym_points)
    gca().set_xticklabels([r'$\Gamma$', 'Y', 'S', 'X', r'$\Gamma$'])
    ylim([0, 20])    # or [0, 55] THz
    gca().set_yticks(linspace(0, 20, 5))
    ylabel(r'$\nu$ (THz)')
    legend(frameon=True, loc="best")
    title("Fivefold3-DNWs")
    #savefig("Fivefold3-DNWs.pdf", bbox_inches='tight')
    show()

if __name__ == "__main__":

    data_file = 'Fivefold3_POSCAR'
    linear_path, sym_points, labels = read_structure(data_file)
    plot_dispersion_gpumd(linear_path, sym_points)

    print('******************** All Done !!! *************************')