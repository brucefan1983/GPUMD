import numpy as np
import matplotlib.pyplot as plt

thermo = np.loadtxt("thermo.out", comments="#")

# run.in uses a 1 fs time step and writes thermo data every 10 steps.
time_ps = np.arange(1, len(thermo) + 1) * 0.01

fig, ax = plt.subplots()
ax.plot(time_ps, thermo[:, 0])
ax.set_xlabel("Time (ps)")
ax.set_ylabel("Temperature (K)")
fig.tight_layout()

fig.savefig("thermo.png", dpi=200)
plt.close(fig)
