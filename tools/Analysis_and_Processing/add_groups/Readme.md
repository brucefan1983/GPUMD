# Atom grouping

`add_groups.py` assigns or updates the `group` array in `model.xyz`.
It requires Python 3, ASE, and NumPy.

Run the required command from a directory containing the script and `model.xyz`.
Each command reads and overwrites `model.xyz`; save a copy before editing it.

| Command | Effect |
| --- | --- |
| `python3 add_groups.py block INF INF INF INF INF 5 4` | Assign atoms with z ≤ 5 Å to group 4. |
| `python3 add_groups.py union 0-3 4 5-6 7` | Merge existing groups 0 through 6 into group 7. |
| `python3 add_groups.py elements` | Assign one group per element, starting at 0. |
| `python3 add_groups.py direction x 8` | Divide the occupied x-coordinate range into 8 groups. |
| `python3 add_groups.py all 0` | Assign all atoms to group 0. |
| `python3 add_groups.py cylinder z 0 0 5 INF INF 1` | Assign atoms within radius 5 Å of the z axis through (0, 0) to group 1. |
| `python3 add_groups.py id` | Assign each atom its own group, starting at 0. |

For `block`, arguments are `xmin xmax ymin ymax zmin zmax group_id`.
For `cylinder`, arguments are `axis center1 center2 radius lo hi group_id`;
the center coordinates lie in the plane perpendicular to the axis.
`INF` leaves the corresponding coordinate bound unrestricted.
Atoms outside a selected region retain their previous groups, or group 0
if the input has no `group` array.

Author: Yuwen Zhang (984307703@qq.com).
