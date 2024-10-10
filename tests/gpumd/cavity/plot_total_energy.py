import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Order: 
# step time q p cavity_pot cavity_kin
# 6: forces

fig, axes = plt.subplots(figsize=(4.3, 4), nrows=3, sharex=True, dpi=200)

steps = {
    0.1: 0,
    0.05: 1,
    0.01: 2
}

for match in Path.cwd().glob('./*/cavity.out'):
    folder = str(match.parent).split('/')[-1]
    print(folder)
    disp = float(folder.replace('d', '').split('s')[0])
    step = float(folder.replace('d', '').split('s')[1])
    data = np.loadtxt(f'{folder}/cavity.out')
    thermo = np.loadtxt(f'{folder}/thermo.out')
    ax = axes[steps[step]]
    U_cav = data[:, 4]
    K_cav = data[:, 5]
    K = thermo[:,1] + K_cav
    U = thermo[:,2] + U_cav
    K -= K[0]
    U -= U[0]
    T = U+K
    ax.plot(T, alpha=0.5, label=f'Disp. {disp}')
    ax.set_ylabel(f'Step: {step}')
#ax.set_yscale('log')
plt.suptitle(r'$E(t)-E(0)$')
axes[2].set_xlabel('steps')
axes[0].legend(loc='best')
plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig('energy.png')


