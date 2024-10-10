import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, fftfreq

# Order: 
# step time q p cavity_pot cavity_kin
# 6: forces
data = np.loadtxt('cavity.out')
print(data.shape)
thermo = np.loadtxt('thermo.out')
q = data[:, 2]
p = data[:, 3]

fig, axes = plt.subplots(figsize=(5, 7), nrows=5, sharex=True, dpi=200)

t = data[:, 1]
ax = axes[0]

ax.plot(t, q, label='q(t)')
ax.plot(t, p, label='p(t)')
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
p2, = ax2.plot(t, K+U, c='g', alpha=0.4, label='Total')
ax2.set_ylabel('Total E (eV)')
ax.legend(loc='best')
ax.set_ylabel('Sys + cav (eV)')
ax.set_xlabel('Time (fs)')

# Set color
ax2.yaxis.label.set_color(p2.get_color())
ax2.spines["right"].set_edgecolor(p2.get_color())
ax2.tick_params(axis='y', colors=p2.get_color())

ax = axes[4]
ax.plot(t, data[:, 6], label='cos')
ax.plot(t, data[:, 7], label='sin')
ax.set_ylabel('Integral')
ax.set_xlabel('Time (fs)')
ax.legend(loc='best')


plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig('cavity.png')

# Plot the FFT of q(t)
fig, ax = plt.subplots(figsize=(3.3, 3), nrows=1, dpi=200)

n = len(q)
# pad with zeros to avoid aliasing
size = 2 ** np.ceil(np.log2(2 * n - 1)).astype("int")
print(n, size)
if size <= 2 * n:
    raise ValueError("Size and n too small, watch out for aliasing!")

Q = rfft(q-q[0], n)
Q = np.abs(Q) ** 2
Q /= size
Q = Q[:n]

dt = t[1]-t[0]
pos = int(n/2)
f = fftfreq(n, dt)[:pos]
Q = Q[:pos]
ax.plot(f, Q)
#ax.set_xscale('log')
ax.set_ylabel('Power spectrum')
ax.set_xlabel('Frequency (1/fs)')

# Second axis
# Define function and its inverse
frominvfstoTHz = lambda x: x*1e3  
fromTHztoinvfs = lambda x: x*1e-3
ax2 = ax.secondary_xaxis("top", functions=(frominvfstoTHz,fromTHztoinvfs))
ax2.set_xlabel('Frequency (THz)')

ax.set_xlim([0, fromTHztoinvfs(250)])
#ax.set_ylim([1e-6, 4e5])
ax.set_yscale('log')



plt.tight_layout()
plt.savefig('cavity_fft.png')

#ax.set_ylabel(r'max($F_{cav}$)')
