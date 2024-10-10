import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pandas import DataFrame

# Order: 
# step time q p cavity_pot cavity_kin
# 6: forces

fig, axes = plt.subplots(figsize=(4.3, 5), nrows=3, sharex=True, dpi=200)

steps = {
    0.1: 0,
    0.05: 1,
    0.01: 2
}

for match in Path.cwd().glob('./*/cavity.out'):
    folder = str(match.parent).split('/')[-1]
    print(folder)
    step = float(folder.replace('dt', ''))
    data = np.loadtxt(f'{folder}/cavity.out')
    t = data[:, 1]
    thermo = np.loadtxt(f'{folder}/thermo.out')
    ax = axes[steps[step]]


    U_cav = data[:, 4]
    K_cav = data[:, 5]
    K = thermo[:,1] + K_cav
    U = thermo[:,2] + U_cav
    K -= K[0]
    U -= U[0]
    T = U+K

    # Compute rolling average
    df = DataFrame({'T': T})
    mean = df.rolling(50).mean()
    ax.plot(t, T, c='cornflowerblue', alpha=0.5)
    ax.plot(t, mean, c='firebrick', label='Rolling mean')
    ax.set_ylabel(f'Step: {step} fs')
    ax.axhline(0, c='k', alpha=0.3, label=r'$\Delta E=0$')
    ax.set_ylim([-0.01, 0.01])
#ax.set_yscale('log')
plt.suptitle(r'$E(t)-E(0)$')
axes[2].set_xlabel('Time (fs)')
axes[0].legend(loc='best')
plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig('energy.png')


