import matplotlib.pyplot as plt
import numpy as np

# constants
font_size = 12

# output file from GPUMD
thermo = np.loadtxt('thermo.out')

# parameters 
num_steps = thermo.shape[0]
time_step = 0.01 # ps (from run.in)
time = time_step * np.arange(num_steps) # ps
num_temps = 10 # number of temperature points (from run.in)
temp = np.linspace(100, 1000, num_temps) # temperatures (K) (from run.in)
num_cells = 10 # number of cells in each direction (from xyz.in)
num_steps_per_temp = int(num_steps / num_temps)
a = thermo[:, 6:8].mean(1) / num_cells # lattice constants (A)
a_ave = a.reshape(num_temps, num_steps_per_temp)
a_ave = a_ave[:, int(num_steps_per_temp / 2) : -1].mean(1)

# plot results
fig, ax = plt.subplots(ncols=2, nrows=2, constrained_layout=True)

# temperature vs time
ax[0, 0].plot(time, thermo[:, 0])
ax[0, 0].set_xlabel('Time (ps)', fontsize=font_size)
ax[0, 0].set_ylabel('Temperature (K)', fontsize=font_size)
ax[0, 0].axis([0, 200, 0, 1100])
ax[0, 0].set_title('(a)')

# pressure vs time
ax[0, 1].plot(time, thermo[:, 3:5].mean(1))
ax[0, 1].set_xlabel('Time (ps)', fontsize=font_size)
ax[0, 1].set_ylabel('Pressure (GPa)', fontsize=font_size)
ax[0, 1].axis([0, 200, -0.1, 0.4])
ax[0, 1].set_title('(b)')

# lattice constant vs time
ax[1, 0].plot(time, a)
ax[1, 0].set_xlabel('Time (ps)', fontsize=font_size)
ax[1, 0].set_ylabel('a (Angstrom)', fontsize=font_size)
ax[1, 0].axis([0, 200, 5.43, 5.48])
ax[1, 0].set_title('(c)')

# thermal expansion
ax[1, 1].scatter(temp, a_ave)
ax[1, 1].set_xlabel('Temperature (K)', fontsize=font_size)
ax[1, 1].set_ylabel('a (Angstrom)', fontsize=font_size)
ax[1, 1].axis([0, 1100, 5.43, 5.48])
ax[1, 1].set_title('(d)')

plt.show()
