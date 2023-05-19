# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:15 2022

@author: hityingph
"""

from thermo.gpumd.data import load_thermo
from pylab import *

aw = 1.5
fs = 18
lw = 2
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
        ax.tick_params(which='both', axis='both', direction='in', right=False, top=False)

time = np.loadtxt("300K/sdc.out")[:, 0][:1000]
vac_300 = np.mean(np.loadtxt("300K/sdc.out")[:, 1:4], axis = 1)
vac_5000 = np.mean(np.loadtxt("5000K/sdc.out")[:, 1:4], axis = 1)
vac_300 = vac_300.reshape((5, 1000)).T
vac_5000 = vac_5000.reshape((5, 1000)).T  

sdc_300 = np.mean(np.loadtxt("300K/sdc.out")[:, 4:], axis = 1)
sdc_5000 = np.mean(np.loadtxt("5000K/sdc.out")[:, 4:], axis = 1)
sdc_300 = sdc_300.reshape((5, 1000)).T
sdc_5000 = sdc_5000.reshape((5, 1000)).T                        
sdc_300_ave = np.mean(sdc_300, axis = 1)
sdc_5000_ave = np.mean(sdc_5000, axis = 1)
sdc_300_std = np.std(sdc_300, axis = 1) / np.sqrt(5)
sdc_5000_std = np.std(sdc_5000, axis = 1) / np.sqrt(5)

figure(figsize=(8, 12))
subplot(2, 1, 1)
set_fig_properties([gca()])
for i in range(5):
    if i == 0:
        plot(time, vac_300[:,i], ls = '--', lw = lw, color = "C0", label = "300 K")
        plot(time, vac_5000[:,i], ls = '-', lw = 1.5, color = "C1", label = "5000 K")
    else:
        plot(time, vac_300[:,i], ls = '--', lw = 1.5, color = "C0")
        plot(time, vac_5000[:,i], ls = '-', lw = 1.5, color = "C1")        
legend()
xlabel(r"Correlation time (ps)")
ylabel(r"VAC ($\mathrm{{\AA}^{2}/ps^{2}}$)") 
title("(a)")

subplot(2, 1, 2)
set_fig_properties([gca()])
for i in range(5):
    if i == 0:
        plot(time, sdc_300[:,i], ls = '--', lw = 1.5, color = "C0", label = "300 K")
        plot(time, sdc_5000[:,i], ls = '-', lw = 1.5, color = "C1", label = "5000 K")
    else:
        plot(time, sdc_300[:,i], ls = '--', lw = 1.5, color = "C0")
        plot(time, sdc_5000[:,i], ls = '-', lw = 1.5, color = "C1")        
legend()
xlabel(r"Correlation time (ps)")
ylabel(r"SDC ($\mathrm{{\AA}^{2}/ps}$)")  
title("(b)")
subplots_adjust(hspace = 0.3)     
savefig("Diffusion.png", bbox_inches='tight')
        
        
    
    
    
    
    
    
    
    
    
    
    
    