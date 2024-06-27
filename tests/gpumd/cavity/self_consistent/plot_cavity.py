import numpy as np
import matplotlib.pyplot as plt

# Order: 
# step time q p cavity_pot cavity_kin
# 6: forces
data = np.loadtxt('cavity.out')

fig, axes = plt.subplots(figsize=(3.3, 5), nrows=3, sharex=True, dpi=200)

t = data[:, 1]
ax = axes[0]
ax.plot(t, data[:, 2], label='q(t)')
ax.plot(t, data[:, 3], label='p(t)')
ax.set_ylabel('Canonical coordinates')
ax.legend(loc='best')

ax = axes[1]
ax.plot(t, data[:, 4], label='Potential')
ax.plot(t, data[:, 5], label='Kinetic')
ax.set_ylabel('Cavity energies')
ax.legend(loc='best')

ax = axes[2]
#ax.plot(t, np.linalg.norm(data[:, 6:], axis=1))
ax.plot(t, np.max(data[:, 6:], axis=1))
ax.set_ylabel(r'max($F_{cav}$) (eV/Å?)')
ax.set_xlabel('Time (fs)')

plt.tight_layout()
plt.savefig('cavity.png')


