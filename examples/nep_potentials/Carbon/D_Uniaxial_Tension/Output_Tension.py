# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:15 2022

@author: hityingph
"""

from thermo.gpumd.data import load_thermo
from pylab import *

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
        ax.tick_params(which='both', axis='both', direction='in', right=False, top=False)

thermo = load_thermo("Run/")
strain = thermo["Lz"]/thermo["Lz"][0] - 1
stress = thermo["Pz"] * -1

output = np.c_[strain, stress]
np.savetxt("./Uniaxial_Tension_Results.txt", output, fmt='%f', delimiter='    ')

        
        
    
    
    
    
    
    
    
    
    
    
    
    