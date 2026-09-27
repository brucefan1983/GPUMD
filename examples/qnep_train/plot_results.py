import numpy as np
import matplotlib.pyplot as plt

energy = np.loadtxt("energy_train.out", comments="#")
force = np.loadtxt("force_train.out", comments="#")
loss = np.loadtxt("loss.out", comments="#")

# Energy parity
fig, ax = plt.subplots()
ax.plot(energy[:, 1], energy[:, 0], ".", markersize=8)
limits = [min(energy.min(axis=0)), max(energy.max(axis=0))]
ax.plot(limits, limits, "--")
ax.set_xlabel("Reference energy (eV/atom)")
ax.set_ylabel("qNEP energy (eV/atom)")
ax.set_aspect("equal", adjustable="box")
fig.tight_layout()

# Force parity
fig, ax = plt.subplots()
reference_force = force[:, 3:6].ravel()
predicted_force = force[:, 0:3].ravel()
ax.plot(reference_force, predicted_force, ".", markersize=4)
limits = [min(reference_force.min(), predicted_force.min()),
          max(reference_force.max(), predicted_force.max())]
ax.plot(limits, limits, "--")
ax.set_xlabel("Reference force (eV/A)")
ax.set_ylabel("qNEP force (eV/A)")
ax.set_aspect("equal", adjustable="box")
fig.tight_layout()

# Training history
fig, ax = plt.subplots()
generation = loss[:, 0]
for column, label in zip(range(1, 6), ["Total", "L1", "L2", "Energy train", "Force train"]):
    ax.loglog(generation, loss[:, column], label=label)
ax.set_xlabel("Generation")
ax.set_ylabel("Loss / RMSE")
ax.legend()
fig.tight_layout()

plt.show()
