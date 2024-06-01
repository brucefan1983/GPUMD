"""
    Purpose:
        Select structures from orginal large reference dataset based on principal component Analysis (PCA) of 
        descriptor space using farthest point sampling. We use the PbTe as a toy example to show how this script
        works, one need to modify the path of reference dataset, nep model, and selected frame number case by case.

    Ref:
        calorine: https://calorine.materialsmodeling.org/tutorials/visualize_descriptor_space_with_pca.html
        https://github.com/bigd4/PyNEP/blob/master/examples/plot_select_structure.py

    Author:
        Penghua Ying <hityingph(at)163.com>
"""

from ase.io import read, write
from pylab import *
from calorine.nep import get_descriptors
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.spatial.distance import cdist

# Farthest Point Sampling
def farthest_point_sampling(points, n_samples):
    n_points = points.shape[0]
    selected_indices = [np.random.randint(n_points)]
    for _ in range(1, n_samples):
        distances = cdist(points, points[selected_indices])
        min_distances = np.min(distances, axis=1)
        next_index = np.argmax(min_distances)
        selected_indices.append(next_index)
    return selected_indices

aw = 2
fs = 16
lw = 2
font = {'size'   : fs}
matplotlib.rc('font', **font)
matplotlib.rc('axes' , linewidth=aw)

def set_fig_properties(ax_list):
    tl = 8
    tw = 2
    tlm = 4
    
    for ax in ax_list:
        ax.tick_params(which='major', length=tl, width=tw)
        ax.tick_params(which='minor', length=tlm, width=tw)
        ax.tick_params(which='both', axis='both', direction='out', right=False, top=False)
        
        
                         
tol = read("../../examples/11_NEP_potential_PbTe/test.xyz", ":") # read orginal larger reference.xyz


descriptors = []
for i, t in tqdm(enumerate(tol)):
    d = get_descriptors(t, model_filename='../../examples/11_NEP_potential_PbTe/nep.txt')  # get descriptors using the pre-trained nep model
    d_mean = np.mean(d, axis=0) # Use average of each atomic descriptors to get structure descriptors
    descriptors.append(d_mean)
    
descriptors = np.array(descriptors)
print(f'Total frame of structures in dataset: {descriptors.shape[0]}')
print(f'Number of descriptor components:  {descriptors.shape[1]}')
pca = PCA(n_components=2)
pc = pca.fit_transform(descriptors)
p0 = pca.explained_variance_ratio_[0]
p1 = pca.explained_variance_ratio_[1]
print(f'Explained variance for component 0: {p0:.2f}')
print(f'Explained variance for component 1: {p1:.2f}')

# Select 25 structures using FPS
n_samples = 25
selected_indices = farthest_point_sampling(pc, n_samples)
selected_structures = [tol[i] for i in selected_indices]
unselected_structures = [t for i, t in enumerate(tol) if i not in selected_indices]

# Save the selected and unselected structures
write('selected_structures.xyz', selected_structures)

figure(figsize=(10, 8))
set_fig_properties([gca()])
scatter(pc[:, 0], pc[:, 1], alpha=0.5, c="C0", label='All structures')
scatter(pc[selected_indices, 0], pc[selected_indices, 1], s=8, color='C1', label='Selected structures')
xlabel('PC1')
ylabel('PC2')
legend()
savefig('FPS.png', bbox_inches='tight')