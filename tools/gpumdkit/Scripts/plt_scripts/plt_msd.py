import sys
import numpy as np
import matplotlib.pyplot as plt

# Function to read data from the given file
def read_data(file_name):
    data = np.loadtxt(file_name)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]

# Determine the input file
#input_file = sys.argv[1] if len(sys.argv) > 1 else './msd.out'
input_file = './msd.out'

# Read the data from the file
time, msd_x, msd_y, msd_z = read_data(input_file)

# Calculate the total MSD
msd_mean = (msd_x + msd_y + msd_z)/3

# Create a plot for the MSD data
plt.figure(figsize=(6, 4.5), dpi=100)
plt.plot(time, msd_x, label='x')
plt.plot(time, msd_y, label='y')
plt.plot(time, msd_z, label='z')
plt.plot(time, msd_mean, label='mean', color='C4')

# Add titles and labels
#plt.title('MSD vs dt')
plt.xlabel('dt (ps)')
plt.ylabel(r'MSD ($\AA^2$)')
plt.legend()
plt.tight_layout()
#plt.grid(True)

if len(sys.argv) > 1 and sys.argv[1] == 'save':
    plt.savefig('msd.png')
else:
    plt.show()