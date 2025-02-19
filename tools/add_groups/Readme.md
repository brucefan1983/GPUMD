# Atom Grouping Script
This script allows you to group atoms in various ways for computational materials science simulations. It supports grouping by region, union of groups, element type, direction, and within a cylinder. Additionally, it can assign a unique group ID to each atom.
### Run example:
    ./add_groups.py block INF INF INF INF INF 5 4
    Group atoms with Z coordinates between INF and 5, assigning them to group 4.
    ./add_groups.py union 0-3 4 5-6 7
    Change groups 0, 1, 2, 3, 4, 5, and 6 to group 7.
    ./add_groups.py elements
    Automatically group atoms by element type, starting group numbers from 0.
    ./add_groups.py direction x 8
    Divide atoms into 8 groups along the x direction.
    ./add_groups.py all 0
    Set the group of all atoms to 0.  
    ./add_groups.py cylinder z 0 0 5 INF INF 1
    Group atoms within a cylinder along the z-axis with radius 5, centered at (0,0), and height from INF to INF, assigning them to group 1.    
    ./add_groups.py id
    Assign a unique group ID to each atom, starting from 0.

