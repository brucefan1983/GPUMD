from ase.io import read
from pathlib import Path
import numpy as np

forces = {}


def uncertainty(forces):
    M = forces.shape[0]
    L = forces.shape[1]
    N = forces.shape[2]
    m = np.zeros((L, N*3))
    m_sq = np.zeros((L, N*3))
    for j in range(M):
        for i in range(N):
            fx = forces[j, :, i, 0]
            fy = forces[j, :, i, 1]
            fz = forces[j, :, i, 2]
            m[:, i + 0 * N] += fx/M
            m[:, i + 1 * N] += fy/M
            m[:, i + 2 * N] += fz/M
            m_sq[:, i + 0 * N] += fx*fx/M
            m_sq[:, i + 1 * N] += fy*fy/M
            m_sq[:, i + 2 * N] += fz*fz/M

    E = np.zeros((L, N*3))
    for i in range(3*N):
        E[:,i] = m_sq[:,i] - m[:,i]*m[:,i]
    u = np.zeros(L)
    for i in range(3*N):
        u += np.sqrt(E[:,i]) / (3*N)
    return u

for xyz in Path.cwd().glob('observer*.xyz'):
    print(xyz)
    observer = str(xyz.name).split('.')[0]
    structures = read(xyz, ':')
    concatenated = np.array([structure.get_forces() for structure in structures])
    forces[observer] = concatenated


forces = np.array([data for data in forces.values()])

F = forces
M = F.shape[0]
L = F.shape[1]
N = F.shape[2]
F = F.reshape((M, L, 3*N))
F_std = np.std(F, axis=0)
print(F_std.shape)

U = np.mean(F_std, axis=1)
print(U.shape)
calculated = np.loadtxt('active.out')

u_alg = uncertainty(forces)

print(U / calculated[:,1])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(calculated[:,0], calculated[:,1], linewidth=2, label='Active')
ax.plot(calculated[:,0], U, linewidth=2, label='Observer')
ax.plot(calculated[:,0], u_alg, linestyle='--', linewidth=2, label='Python')
ax.legend(loc='best')
ax.set_xlabel('Time (fs)')
ax.set_ylabel(r'Uncertainty $\sigma_f$ (eV/Å/atom)')
plt.show()


