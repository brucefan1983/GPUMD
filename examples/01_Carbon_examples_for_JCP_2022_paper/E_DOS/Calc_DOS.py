from pylab import *
from thermo.gpumd.data import load_dos, load_vac

aw = 2
fs = 16
font = {'size'   : fs}
matplotlib.rc('font', **font)
matplotlib.rc('axes' , linewidth=aw)

def set_fig_properties(ax_list):
    tl = 8
    tw = 2
    tlm = 4
    
    for ax in ax_list:
        ax.tick_params(which='major', length=tl, width=tw)
        ax.tick_params(which='minor', length=tlm, width=tw)
        ax.tick_params(which='both', axis='both', direction='in', right=True, top=True)
        
Nc = 200
dos = load_dos(num_dos_points=Nc, directory="Run/")['run0']
vac = load_vac(Nc, directory="Run/")['run0']
dos['DOSxyz'] = dos['DOSx']+dos['DOSy']+dos['DOSz']
vac['VACxyz'] = vac['VACx']+vac['VACy']+vac['VACz']
vac['VACxyz'] /= vac['VACxyz'].max() 
print('DOS:', dos.keys())
print('VAC:', vac.keys())

figure(figsize=(12,5))
subplot(1,2,1)
set_fig_properties([gca()])
plot(dos['nu'], dos['DOSxyz'], color='k',linewidth=3)
xlim([0, 60])
gca().set_xticks(range(0,61,20))
ylim([0, 6000])
gca().set_yticks(np.arange(0,6001,2000))
ylabel('VDOS (1/THz)')
xlabel(r'$\omega$/2$\pi$ (THz)')
title('(a)')

Temp = np.arange(10,5001,100)  # [K]
N = 64000  # Number of atoms
Cxyz = list() # [k_B/atom] Heat capacity per atom
hnu = 6.63e-34*dos['nu']*1.e12  # [J]
for T in Temp:
    kBT = 1.38e-23*T  # [J]
    x = hnu/kBT
    expr = np.square(x)*np.exp(x)/(np.square(np.expm1(x)))
    Cxyz.append(np.trapz(dos['DOSxyz']*expr, dos['nu'])/N)
subplot(1,2,2)
set_fig_properties([gca()])
mew, ms, mfc, lw = 1, 8, 'none', 2.5
plot(Temp, Cxyz, lw=lw)
xlim([0,5100])
gca().set_xticks(np.linspace(0, 5000, 6))
ylim([0, 4])
gca().set_yticks(np.linspace(0, 4, 5))
ylabel(r'Heat Capacity (k$_B$/atom)')
xlabel('Temperature (K)')
title('(b)')
tight_layout()
show()    
savefig("DOS.png", bbox_inches='tight')

output1 = np.c_[dos['nu'], dos['DOSxyz']]
output2 = np.c_[Temp, Cxyz]
np.savetxt("./DOS_results.txt", output1, fmt='%f', delimiter='    ')
np.savetxt("./HeatCapacity_results.txt", output2, fmt='%f', delimiter='    ')


