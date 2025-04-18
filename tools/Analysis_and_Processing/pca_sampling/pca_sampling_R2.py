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
        Zherui Chen <chenzherui0124@foxmail.com>
"""


from ase.io import read, write
from pylab import *
from calorine.nep import get_descriptors
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist

# -------------------------------------------------------------------
# Input/Output and Configuration
# -------------------------------------------------------------------
INPUT_FILE = "train.xyz"               # Input file containing structures
NEP_MODEL_FILE = "nep.txt"             # Input NEP model file for descriptor generation
OUTPUT_SELECTED_FILE = "selected_structures.xyz"  # Output file for selected structures
OUTPUT_UNSELECTED_FILE = "unselected_structures.xyz"  # Output file for unselected structures
OUTPUT_PLOT_FILE = "FPS.png"           # Output file for visualization

R2_THRESHOLD = 0.90  # R2 threshold to stop sampling
PCA_COMPONENTS = 2   # Number of PCA components to use for dimensionality reduction
# -------------------------------------------------------------------

# Incremental Farthest Point Sampling with early stopping
def incremental_fps_with_r2(pc, r2_threshold):
    n_points = pc.shape[0]
    overall_mean = np.mean(pc, axis=0)  # Mean of all points
    total_variance = np.sum((pc - overall_mean) ** 2)  # Total variance of data

    # Randomly select the first point
    selected_indices = [np.random.randint(n_points)]
    min_distances = np.full(n_points, np.inf)  # Initialize distances to infinity

    explained_variance = 0
    r2 = 0.0

    # Loop to incrementally select points
    for _ in range(1, n_points):
        # Update min_distances: minimum distance from each point to the selected points
        new_point_idx = selected_indices[-1]
        new_distances = cdist(pc, pc[[new_point_idx]]).flatten()
        min_distances = np.minimum(min_distances, new_distances)

        # Select the next farthest point
        next_index = np.argmax(min_distances)
        selected_indices.append(next_index)

        # Calculate explained variance incrementally
        explained_variance = np.sum((pc[selected_indices] - overall_mean) ** 2)
        r2 = explained_variance / total_variance

        # Early stopping if R2 exceeds threshold
        if r2 >= r2_threshold:
            break

    return selected_indices, r2


# Set figure properties for plotting
def set_fig_properties(ax_list):
    tl = 8
    tw = 2
    tlm = 4
    for ax in ax_list:
        ax.tick_params(which="major", length=tl, width=tw)
        ax.tick_params(which="minor", length=tlm, width=tw)
        ax.tick_params(which="both", axis="both", direction="out", right=False, top=False)


# -------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------

# Load dataset
tol = read(INPUT_FILE, ":")  # Read original larger reference.xyz

# Generate descriptors
descriptors = []
for i, t in tqdm(enumerate(tol)):
    d = get_descriptors(t, model_filename=NEP_MODEL_FILE)  # Use NEP model file as input for descriptors
    d_mean = np.mean(d, axis=0)  # Use average of each atomic descriptor to get structure descriptors
    descriptors.append(d_mean)

descriptors = np.array(descriptors)
print(f"Total number of structures in dataset: {descriptors.shape[0]}")
print(f"Number of descriptor components: {descriptors.shape[1]}")

# PCA
pca = PCA(n_components=PCA_COMPONENTS)
pc = pca.fit_transform(descriptors)
p0 = pca.explained_variance_ratio_[0]
p1 = pca.explained_variance_ratio_[1]
print(f"Explained variance for component 0: {p0:.2f}")
print(f"Explained variance for component 1: {p1:.2f}")

# Find minimum samples to achieve the threshold
selected_indices, final_r2 = incremental_fps_with_r2(pc, R2_THRESHOLD)

# Separate selected and unselected structures
selected_structures = [tol[i] for i in selected_indices]
unselected_structures = [t for i, t in enumerate(tol) if i not in selected_indices]

# Save the selected and unselected structures
write(OUTPUT_SELECTED_FILE, selected_structures)
write(OUTPUT_UNSELECTED_FILE, unselected_structures)

# Visualization
figure(figsize=(10, 8))
set_fig_properties([gca()])
scatter(pc[:, 0], pc[:, 1], alpha=0.5, c="C0", label="All structures")
scatter(pc[selected_indices, 0], pc[selected_indices, 1], s=8, color="C1", label="Selected structures")
xlabel("PC1")
ylabel("PC2")
legend()

# Add final R2 and number of samples to the plot
text(
    0.05,
    0.95,
    f"R$^2$: {final_r2:.3f}\nSamples: {len(selected_indices)}",
    transform=gca().transAxes,
    fontsize=14,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
)

savefig(OUTPUT_PLOT_FILE, bbox_inches="tight")