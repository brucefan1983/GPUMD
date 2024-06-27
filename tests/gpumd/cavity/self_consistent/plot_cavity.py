import numpy as np
import matplotlib.pyplot as plt

# Order: 
# step time q p cavity_pot cavity_kin
# 6: forces
data = np.loadtxt('cavity.out')
thermo = np.loadtxt('thermo.out')

fig, axes = plt.subplots(figsize=(5, 4), nrows=4, sharex=True, dpi=200)

t = data[:, 1]
ax = axes[0]
ax.plot(t, data[:, 2], label='q(t)')
ax.plot(t, data[:, 3], label='p(t)')
ax.set_ylabel('Cav. coord')
ax.legend(loc='best')

ax = axes[1]
U_cav = data[:, 4]
K_cav = data[:, 5]
ax.plot(t, U_cav, label='Potential')
ax.plot(t, K_cav, label='Kinetic')
ax.set_ylabel('Cav. E')
ax.legend(loc='best')

ax = axes[2]
#ax.plot(t, np.linalg.norm(data[:, 6:], axis=1))
ax.plot(t, np.max(data[:, 6:], axis=1))
ax.set_ylabel(r'max($F_{cav}$)')

ax = axes[3]
K = thermo[:,1] + K_cav
U = thermo[:,2] + U_cav
K -= K[0]
U -= U[0]
ax.plot(t, U, label='Potential')
ax.plot(t, K, label='Kinetic')
ax2 = ax.twinx()
p2, = ax2.plot(t, K+U, c='g', label='Total')
ax2.set_ylabel('Total E (eV)')
ax.legend(loc='best')
ax.set_ylabel('System + cav (eV)')
ax.set_xlabel('Time (fs)')

# Set color
ax2.yaxis.label.set_color(p2.get_color())
ax2.spines["right"].set_edgecolor(p2.get_color())
ax2.tick_params(axis='y', colors=p2.get_color())

plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig('cavity.png')


