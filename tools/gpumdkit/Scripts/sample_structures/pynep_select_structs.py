import sys
import numpy as np
from ase.io import read, write
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pynep.calculate import NEP
from pynep.select import FarthestPointSample

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

# Check command line arguments
if len(sys.argv) < 5:
    print(" Usage: python pynep_select_structs.py <sampledata_file> <traindata_file> <nep_model_file> <min_distance>")
    print(" Examp: python pynep_select_structs.py dump.xyz train.xyz ./nep.txt 0.01")
    sys.exit(1)

# Load data
sampledata = read(sys.argv[1], ':')
traindata = read(sys.argv[2], ':')

# Initialize NEP calculator
calc = NEP(sys.argv[3])
print(calc)

# Calculate descriptors with progress bar
total_sample = len(sampledata)
total_train = len(traindata)

des_sample = []
for i in range(total_sample):
    des_sample.append(np.mean(calc.get_property('descriptor', sampledata[i]), axis=0))
    print_progress_bar(i + 1, total_sample, prefix=' Processing sampledata:', suffix='Complete', length=50)
#des_sample = np.load('des_sample.npy')
des_sample = np.array(des_sample)
#np.save('des_sample.npy', des_sample)

des_train = []
for i in range(total_train):
    des_train.append(np.mean(calc.get_property('descriptor', traindata[i]), axis=0))
    print_progress_bar(i + 1, total_train, prefix=' Processing traindata: ', suffix='Complete', length=50)
#des_train = np.load('des_train.npy')
des_train = np.array(des_train)
#np.save('des_train.npy', des_train)

# Farthest Point Sampling
min_dist = float(sys.argv[4])
sampler = FarthestPointSample(min_distance=min_dist)
selected = sampler.select(des_sample, des_train)
write('selected.xyz', [sampledata[i] for i in selected])

# Check if seaborn is installed
try:
    import seaborn as sns
    sns_installed = True
except ImportError:
    sns_installed = False

# PCA for dimensionality reduction and visualization
reducer = PCA(n_components=2)
reducer.fit(des_sample)
proj_sample = reducer.transform(des_sample)
proj_train = reducer.transform(des_train)
proj_selected = reducer.transform(np.array([des_sample[i] for i in selected]))

# Create the figure
plt.figure(figsize=(5, 5), dpi=200)

# Add the main scatter plot
main_ax = plt.gca()
main_ax.scatter(proj_sample[:, 0], proj_sample[:, 1], color='C0', label=sys.argv[1], alpha=0.4)
main_ax.scatter(proj_train[:, 0], proj_train[:, 1], color='C1', label=sys.argv[2], alpha=0.4)
main_ax.scatter(proj_selected[:, 0], proj_selected[:, 1], color='C2', label='selected.xyz', alpha=0.4)
main_ax.set_xlabel('PC1')
main_ax.set_ylabel('PC2')
main_ax.set_xticks([])
main_ax.set_yticks([])
main_ax.legend()

# Add projections if seaborn is available
if sns_installed:
    # Add density plots
    top_kde = main_ax.inset_axes([0, 1.05, 1, 0.2], transform=main_ax.transAxes)
    sns.kdeplot(x=proj_sample[:, 0], color='C0', ax=top_kde, fill=True, alpha=0.4)
    sns.kdeplot(x=proj_train[:, 0], color='C1', ax=top_kde, fill=True, alpha=0.4)
    sns.kdeplot(x=proj_selected[:, 0], color='C2', ax=top_kde, fill=True, alpha=0.4)
    top_kde.set_xticks([])
    top_kde.set_yticks([])

    side_kde = main_ax.inset_axes([1.05, 0, 0.2, 1], transform=main_ax.transAxes)
    sns.kdeplot(y=proj_sample[:, 1], color='C0', ax=side_kde, fill=True, alpha=0.4)
    sns.kdeplot(y=proj_train[:, 1], color='C1', ax=side_kde, fill=True, alpha=0.4)
    sns.kdeplot(y=proj_selected[:, 1], color='C2', ax=side_kde, fill=True, alpha=0.4)
    side_kde.set_xticks([])
    side_kde.set_yticks([])

plt.tight_layout()
#plt.show()
plt.savefig('select.png')