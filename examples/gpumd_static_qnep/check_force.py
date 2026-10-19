import numpy as np
import matplotlib.pyplot as plt

force_gpumd = np.loadtxt("dump.xyz", skiprows=2, usecols=(4, 5, 6))
force_nep = np.loadtxt("../qnep_train/force_train.out", comments="#")[-len(force_gpumd):, :3]
difference = force_gpumd - force_nep

print(f"Maximum absolute force difference: {np.max(np.abs(difference)):.3e} eV/A")
print(f"RMS force difference: {np.sqrt(np.mean(difference**2)):.3e} eV/A")

fig, ax = plt.subplots()
ax.plot(difference)
ax.set_xlabel("Atom index")
ax.set_ylabel("Force difference (eV/A)")
fig.tight_layout()

fig.savefig("force_difference.png", dpi=200)
plt.close(fig)
