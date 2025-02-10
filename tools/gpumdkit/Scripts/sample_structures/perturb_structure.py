import sys
import dpdata

# Define default values for parameters
DEFAULT_PERT_NUM = 20
DEFAULT_CELL_PERT_FRACTION = 0.03
DEFAULT_ATOM_PERT_DISTANCE = 0.2
DEFAULT_ATOM_PERT_STYLE = 'uniform'

# Check command line arguments
if len(sys.argv) < 2:
    print("Usage: python script.py <input.vasp> <pert_num> <cell_pert_fraction> <atom_pert_distance> <atom_pert_style>")
    print(f"Default values: pert_num={DEFAULT_PERT_NUM}, cell_pert_fraction={DEFAULT_CELL_PERT_FRACTION}, atom_pert_distance={DEFAULT_ATOM_PERT_DISTANCE}, atom_pert_style={DEFAULT_ATOM_PERT_STYLE}")
    print("atom_pert_style options: 'normal', 'uniform', 'const'")
    print("dpdata documentation: https://docs.deepmodeling.com/projects/dpdata/en/master/index.html")
    sys.exit(1)

# Read command line arguments
input_file = sys.argv[1]

# Set default values
pert_num = DEFAULT_PERT_NUM
cell_pert_fraction = DEFAULT_CELL_PERT_FRACTION
atom_pert_distance = DEFAULT_ATOM_PERT_DISTANCE
atom_pert_style = DEFAULT_ATOM_PERT_STYLE

# Override default values with user inputs if provided
if len(sys.argv) > 2:
    pert_num = int(sys.argv[2])
if len(sys.argv) > 3:
    cell_pert_fraction = float(sys.argv[3])
if len(sys.argv) > 4:
    atom_pert_distance = float(sys.argv[4])
if len(sys.argv) > 5:
    atom_pert_style = sys.argv[5]

# Read the POSCAR file and perform perturbation
system = dpdata.System(input_file, fmt='vasp/poscar')
perturbed_systems = system.perturb(
    pert_num=pert_num,
    cell_pert_fraction=cell_pert_fraction,
    atom_pert_distance=atom_pert_distance,
    atom_pert_style=atom_pert_style,
)

# Save the perturbed structures
for i in range(pert_num):
    output_file = f"POSCAR_{i + 1}.vasp"
    perturbed_systems.sub_system(i).to('vasp/poscar', output_file)

