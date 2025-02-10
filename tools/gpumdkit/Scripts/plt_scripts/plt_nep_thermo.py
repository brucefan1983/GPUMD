import sys
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('thermo.out')

dump_interval = 10  
time = np.arange(0, len(data) * dump_interval / 1000, dump_interval / 1000)

# read data
temperature = data[:, 0]
kinetic_energy = data[:, 1]
potential_energy = data[:, 2]
pressure_x = data[:, 3]
pressure_y = data[:, 4]
pressure_z = data[:, 5]

num_columns = data.shape[1]

if num_columns == 12:
    box_length_x = data[:, 9]
    box_length_y = data[:, 10]
    box_length_z = data[:, 11]
elif num_columns == 18:
    ax, ay, az = data[:, 9], data[:, 10], data[:, 11]
    bx, by, bz = data[:, 12], data[:, 13], data[:, 14]
    cx, cy, cz = data[:, 15], data[:, 16], data[:, 17]
    
    box_length_x = np.sqrt(ax**2 + ay**2 + az**2)
    box_length_y = np.sqrt(bx**2 + by**2 + bz**2)
    box_length_z = np.sqrt(cx**2 + cy**2 + cz**2)
else:
    raise ValueError("Unsupported number of columns in thermo.out. Expected 12 or 18.")

# subplot
fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), dpi=100)

# Temperature
axs[0, 0].plot(time, temperature)
axs[0, 0].set_title('Temperature')
axs[0, 0].set_xlabel('Time (ps)')
axs[0, 0].set_ylabel('Temperature (K)')

# Potential Energy and Kinetic Energy
color_potential = 'tab:orange'
color_kinetic = 'tab:green'
axs[0, 1].set_title(r'$P_E$ vs $K_E$')
axs[0, 1].set_xlabel('Time (ps)')
axs[0, 1].set_ylabel('Potential Energy (eV)', color=color_potential)
axs[0, 1].plot(time, potential_energy, color=color_potential)
axs[0, 1].tick_params(axis='y', labelcolor=color_potential)

axs_kinetic = axs[0, 1].twinx()
axs_kinetic.set_ylabel('Kinetic Energy (eV)', color=color_kinetic)
axs_kinetic.plot(time, kinetic_energy, color=color_kinetic)
axs_kinetic.tick_params(axis='y', labelcolor=color_kinetic)

# Pressure
axs[1, 0].plot(time, pressure_x, label='Px')
axs[1, 0].plot(time, pressure_y, label='Py')
axs[1, 0].plot(time, pressure_z, label='Pz')
axs[1, 0].set_title('Pressure')
axs[1, 0].set_xlabel('Time (ps)')
axs[1, 0].set_ylabel('Pressure (GPa)')
axs[1, 0].legend()

# Lattice
axs[1, 1].plot(time, box_length_x, label='Lx')
axs[1, 1].plot(time, box_length_y, label='Ly')
axs[1, 1].plot(time, box_length_z, label='Lz')
axs[1, 1].set_title('Lattice Parameters')
axs[1, 1].set_xlabel('Time (ps)')
axs[1, 1].set_ylabel(r'Lattice Parameters ($\AA$)')
axs[1, 1].legend()

plt.tight_layout()

if len(sys.argv) > 1 and sys.argv[1] == 'save':
    plt.savefig('thermo.png')
else:
    plt.show()