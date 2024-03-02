# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 21:27:04 2022

@author: hityingph
"""

from thermo.gpumd.data import load_thermo
from pylab import *
import math
import pandas as pd

aw = 1.5
fs = 14
font = {'size'   : fs}
matplotlib.rc('font', **font)
matplotlib.rc('axes' , linewidth=aw)

def set_fig_properties(ax_list):
    tl = 6
    tw = 1.5
    tlm = 3
    
    for ax in ax_list:
        ax.tick_params(which='major', length=tl, width=tw)
        ax.tick_params(which='minor', length=tlm, width=tw)
        ax.tick_params(which='both', axis='both', direction='in', right=True, top=True)

# grid = plt.GridSpec(3, 1, hspace=0.2)

figure(figsize=(10, 8))
subplot(2,2,1)
set_fig_properties([gca()])
thermo = load_thermo("Run/")
time1 = np.arange(1, len(thermo['T']) + 1) * 10 / 1000 #unit in ps
temp = thermo['T']
plot(time1, temp, lw = 2.0, color = "blue")
vlines(30, -2000, 20000, colors = "black", lw = 1.5, ls = "--")
vlines(60, -2000, 20000, colors = "black", lw = 1.5, ls = "--")
vlines(60.5, -2000, 20000, colors = "black", lw = 1.5, ls = "--")
vlines(90.5, -2000, 20000, colors = "black", lw = 1.5, ls = "--")
vlines(120.5, -2000, 20000, colors = "black", lw = 1.5, ls = "--")
xlim([0, 120])
gca().set_xticks(linspace(0, 120, 7))
ylim([0, 9300])
gca().set_yticks([300, 1000, 5000, 9000])
xlabel('Simulation Time (ps)')
ylabel(r'$T$ (K)')
title(r'(a)')

subplot(2,2,2)
set_fig_properties([gca()])
sp3_ratio = np.loadtxt("Quench_Sp3Ratio_Results.txt")
time2 = np.zeros(len(sp3_ratio))
for i in range(len(sp3_ratio)):
    if i < 60:
        time2[i] = i + 1  #unit in ps
    elif i < 65:
        time2[i] = 61  + (i - 60) * 100 / 1000  #unit in ps
    else:
        time2[i] = 61.5 + (i - 65)  #unit in ps 
plot(time2, sp3_ratio*100, lw = 2.0, color = "red")
vlines(30, -2000, 20000, colors = "black", lw = 1.5, ls = "--")
vlines(60, -2000, 20000, colors = "black", lw = 1.5, ls = "--")
vlines(60.5, -2000, 20000, colors = "black", lw = 1.5, ls = "--")
vlines(90.5, -2000, 20000, colors = "black", lw = 1.5, ls = "--")
vlines(120.5, -2000, 20000, colors = "black", lw = 1.5, ls = "--")
xlim([0, 120])
gca().set_xticks(linspace(0, 120, 7))
ylim([25, 65])
gca().set_yticks([20, 30, 40, 50, 60])
xlabel('Simulation Time (ps)')
ylabel('sp$_{3}$ atoms ratio (%)')
output1 = np.c_[time1, temp]
output2 = np.c_[time2, sp3_ratio]
np.savetxt("./Quench_Thermo_a.txt", output1, fmt='%f', delimiter='    ')
np.savetxt("./Quench_Thermo_b.txt", output2, fmt='%f', delimiter='    ')
title(r'(b)')

subplot(2,2,3)
set_fig_properties([gca()])
rdf_300K = np.loadtxt("avg_RDF-300K.txt")
rdf_5000K = np.loadtxt("avg_RDF-5000K.txt")
plot(rdf_5000K[:,0], rdf_5000K[:,1], lw = 2.0, color = "red", label = "5000 K")
plot(rdf_300K[:,0], rdf_300K[:,1], lw = 2.0, color = "blue", label = "300 K")
legend()
gca().set_yticks([])
xlabel(r'r ($\rm \AA$)')
ylabel(r'g(r)')
title(r'(c)')

subplot(2,2,4)
set_fig_properties([gca()])
adf_300K = np.loadtxt("avg_bond_angles-300K.txt")
adf_5000K = np.loadtxt("avg_bond_angles-5000K.txt")
plot(adf_5000K[:,0], adf_5000K[:,1], lw = 2.0, color = "red", label = "5000 K")
plot(adf_300K[:,0], adf_300K[:,1], lw = 2.0, color = "blue", label = "300 K")
legend()
gca().set_yticks([])
xlabel(r'$\theta$ (deg)')
ylabel(r'g($\theta$)')
title(r'(d)')

subplots_adjust(hspace = 0.4, wspace = 0.3)
savefig("Quench_thermo.eps", bbox_inches='tight')


    




    
    
    
    
    
