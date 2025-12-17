from __future__ import annotations
from dataclasses import dataclass
import os, shutil
import numpy as np

MASS_TABLE = {
    "H": 1.0080000000,
    "He": 4.0026020000,
    "Li": 6.9400000000,
    "Be": 9.0121831000,
    "B": 10.8100000000,
    "C": 12.0110000000,
    "N": 14.0070000000,
    "O": 15.9990000000,
    "F": 18.9984031630,
    "Ne": 20.1797000000,
    "Na": 22.9897692800,
    "Mg": 24.3050000000,
    "Al": 26.9815385000,
    "Si": 28.0850000000,
    "P": 30.9737619980,
    "S": 32.0600000000,
    "Cl": 35.4500000000,
    "Ar": 39.9480000000,
    "K": 39.0983000000,
    "Ca": 40.0780000000,
    "Sc": 44.9559080000,
    "Ti": 47.8670000000,
    "V": 50.9415000000,
    "Cr": 51.9961000000,
    "Mn": 54.9380440000,
    "Fe": 55.8450000000,
    "Co": 58.9331940000,
    "Ni": 58.6934000000,
    "Cu": 63.5460000000,
    "Zn": 65.3800000000,
    "Ga": 69.7230000000,
    "Ge": 72.6300000000,
    "As": 74.9215950000,
    "Se": 78.9710000000,
    "Br": 79.9040000000,
    "Kr": 83.7980000000,
    "Rb": 85.4678000000,
    "Sr": 87.6200000000,
    "Y": 88.9058400000,
    "Zr": 91.2240000000,
    "Nb": 92.9063700000,
    "Mo": 95.9500000000,
    "Tc": 98.0000000000,
    "Ru": 101.0700000000,
    "Rh": 102.9055000000,
    "Pd": 106.4200000000,
    "Ag": 107.8682000000,
    "Cd": 112.4140000000,
    "In": 114.8180000000,
    "Sn": 118.7100000000,
    "Sb": 121.7600000000,
    "Te": 127.6000000000,
    "I": 126.9044700000,
    "Xe": 131.2930000000,
    "Cs": 132.9054519600,
    "Ba": 137.3270000000,
    "La": 138.9054700000,
    "Ce": 140.1160000000,
    "Pr": 140.9076600000,
    "Nd": 144.2420000000,
    "Pm": 145.0000000000,
    "Sm": 150.3600000000,
    "Eu": 151.9640000000,
    "Gd": 157.2500000000,
    "Tb": 158.9253500000,
    "Dy": 162.5000000000,
    "Ho": 164.9303300000,
    "Er": 167.2590000000,
    "Tm": 168.9342200000,
    "Yb": 173.0450000000,
    "Lu": 174.9668000000,
    "Hf": 178.4900000000,
    "Ta": 180.9478800000,
    "W": 183.8400000000,
    "Re": 186.2070000000,
    "Os": 190.2300000000,
    "Ir": 192.2170000000,
    "Pt": 195.0840000000,
    "Au": 196.9665690000,
    "Hg": 200.5920000000,
    "Tl": 204.3800000000,
    "Pb": 207.2000000000,
    "Bi": 208.9804000000,
    "Po": 210.0000000000,
    "At": 210.0000000000,
    "Rn": 222.0000000000,
    "Fr": 223.0000000000,
    "Ra": 226.0000000000,
    "Ac": 227.0000000000,
    "Th": 232.0377000000,
    "Pa": 231.0358800000,
    "U": 238.0289100000,
    "Np": 237.0000000000,
    "Pu": 244.0000000000,
    "Am": 243.0000000000,
    "Cm": 247.0000000000,
    "Bk": 247.0000000000,
    "Cf": 251.0000000000,
    "Es": 252.0000000000,
    "Fm": 257.0000000000,
    "Md": 258.0000000000,
    "No": 259.0000000000,
    "Lr": 262.0000000000,
}


@dataclass
class _AtomView:
    """Lightweight view over a single atom within a Structure."""

    _structure: "Structure"
    index: int

    @property
    def species(self):
        return self._structure.species[self.index]

    @species.setter
    def species(self, value):
        species_str = str(value)
        self._structure.species[self.index] = species_str
        # Update mass to match new species when possible.
        self._structure.masses[self.index] = MASS_TABLE.get(
            species_str, self._structure.masses[self.index]
        )

    @property
    def position(self):
        return tuple(self._structure.positions[self.index])

    @position.setter
    def position(self, value):
        self._structure.positions[self.index] = np.asarray(value, dtype=float)

    @property
    def mass(self):
        return float(self._structure.masses[self.index])

    @mass.setter
    def mass(self, value):
        self._structure.masses[self.index] = float(value)

    @property
    def velocity(self):
        if self._structure.velocities is None:
            return None
        return tuple(self._structure.velocities[self.index])
    
    @velocity.setter
    def velocity(self, value):
        if self._structure.velocities is None:
            raise AttributeError("Structure does not store velocities")
        self._structure.velocities[self.index] = np.asarray(value, dtype=float)

    @property
    def groups(self):
        if self._structure.groups is None:
            return None
        return [int(group[self.index]) for group in self._structure.groups]
    
    @groups.setter
    def groups(self, value):
        if self._structure.groups is None:
            raise AttributeError("Structure does not store groups")
        if len(value) != len(self._structure.groups):
            raise ValueError("Number of group values must match number of groups in Structure")
        for i, group_value in enumerate(value):
            self._structure.groups[i][self.index] = int(group_value)


class Structure:
    """Class to hold atomic structure information.
    
    Parameters
    ----------
    species : list[str]
        Element species as list of strings.
    positions : list[list[float]] or np.ndarray of shape (N, 3)
        Accepts list or array input; stored as float ndarray.
    masses : list[float] or np.ndarray of shape (N,)
        Accepts list or array input; inferred from MASS_TABLE when omitted.
    velocities : list[list[float]] or np.ndarray of shape (N, 3)
        Accepts list or array input; stored as float ndarray.
    groups : list[list[int]] or list[np.ndarray]
        Each entry is length-N labels; accepts list or array input per group.
    lattice : list[float] or np.ndarray of shape (3, 3)
        Accepts nine-value list or array; stored as 3x3 float ndarray.
    pbc : tuple[bool, bool, bool]
        Defaults to (True, True, True).
    info : dict
        Optional metadata dictionary.
    """

    def __init__(self, species=None, positions=None, lattice=None,
                 pbc=None, masses=None, velocities=None,
                 groups=None, info=None):
        self._set_species(species)
        self._set_positions(positions)
        self._set_lattice(lattice)
        self._set_pbc(pbc)
        self._set_masses(masses)
        self._set_velocities(velocities)
        self._set_groups(groups)
        self.set_info(info)

    def __len__(self):
        return self.N

    def __iter__(self):
        for index in range(len(self)):
            yield _AtomView(self, index)

    def __getitem__(self, index):
        natoms = len(self)
        if index < 0:
            index += natoms
        if not 0 <= index < natoms:
            raise IndexError("atom index out of range")
        return _AtomView(self, index)

    def add_atom(self, species, position, mass=None, velocity=None, group_idxs=None):
        species_str = str(species)
        position_array = np.asarray(position, dtype=float)
        if position_array.shape != (3,):
            raise ValueError("position must contain three values")

        mass_value = float(mass) if mass is not None else float(MASS_TABLE.get(species_str, 0.0))

        self.species.append(species_str)
        self.positions = np.vstack([self.positions, position_array])
        self.masses = np.append(self.masses, mass_value)

        if self.velocities is None:
            if velocity is not None:
                raise ValueError("structure does not store velocities; cannot add velocity data")
        else:
            if velocity is None:
                raise ValueError("velocity values are required for structures storing velocities")
            velocity_array = np.asarray(velocity, dtype=float)
            if velocity_array.shape != (3,):
                raise ValueError("velocity must contain three values")
            self.velocities = np.vstack([self.velocities, velocity_array])

        if self.groups is None:
            if group_idxs is not None:
                raise ValueError("structure does not store groups; cannot add group data")
        else:
            if group_idxs is None:
                raise ValueError("group values are required for structures storing groups")
            if len(group_idxs) != len(self.groups):
                raise ValueError("group_idxs length must match number of stored groups")
            for idx, value in enumerate(group_idxs):
                self.groups[idx] = np.append(self.groups[idx], int(value))

        self.N += 1
        return self.N - 1

    def append(self, species, position, mass=None, velocity=None, group_idxs=None):
        """Alias for add_atom to mimic list-style append."""
        return self.add_atom(species, position, mass=mass, velocity=velocity, group_idxs=group_idxs)

    def delete(self, index):
        """Remove the atom at the given index."""
        natoms = len(self)
        if index < 0:
            index += natoms
        if not 0 <= index < natoms:
            raise IndexError("atom index out of range")

        del self.species[index]
        self.positions = np.delete(self.positions, index, axis=0)
        self.masses = np.delete(self.masses, index, axis=0)

        if self.velocities is not None:
            self.velocities = np.delete(self.velocities, index, axis=0)

        if self.groups is not None:
            self.groups = [np.delete(group, index, axis=0) for group in self.groups]

        self.N -= 1

    def repeat(self, reps):
        """Create a supercell by repeating the structure along each lattice vector."""
        nx, ny, nz = self._parse_reps(reps)
        total_copies = nx * ny * nz

        a1, a2, a3 = self.lattice
        offsets = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    offsets.append(ix * a1 + iy * a2 + iz * a3)
        offsets = np.asarray(offsets, dtype=float)

        repeated_positions = np.vstack([self.positions + offset for offset in offsets])

        repeated_species = []
        for _ in range(total_copies):
            repeated_species.extend(self.species)

        repeated_masses = np.tile(self.masses, total_copies)

        if self.velocities is None:
            repeated_velocities = None
        else:
            repeated_velocities = np.tile(self.velocities, (total_copies, 1))

        if self.groups is None:
            repeated_groups = None
        else:
            repeated_groups = [np.tile(group, total_copies) for group in self.groups]

        new_lattice = self.lattice.copy()
        new_lattice[0] *= nx
        new_lattice[1] *= ny
        new_lattice[2] *= nz

        return Structure(
            species=repeated_species,
            positions=repeated_positions,
            lattice=new_lattice,
            pbc=self.get_pbc(),
            masses=repeated_masses,
            velocities=repeated_velocities,
            groups=repeated_groups,
            info=self.info.copy(),
        )

    def __mul__(self, reps):
        return self.repeat(reps)

    def __rmul__(self, reps):
        return self.repeat(reps)

    def _parse_reps(self, reps):
        if isinstance(reps, int):
            rep_values = (reps, reps, reps)
        elif isinstance(reps, (list, tuple, np.ndarray)):
            if len(reps) != 3:
                raise ValueError("repetition factors must contain three integers")
            rep_values = tuple(int(value) for value in reps)
        else:
            raise TypeError("repetition factors must be int, list, tuple, or numpy array")
        if any(value <= 0 for value in rep_values):
            raise ValueError("repetition factors must be positive integers")
        return rep_values

    def _set_species(self, species):
        if species is None:
            raise ValueError("species must be provided")
        if not isinstance(species, list):
            raise TypeError("species must be a list of strings")
        self.species = [str(item) for item in species]
        self.N = len(self.species)

    def _set_positions(self, positions):
        if positions is None:
            raise ValueError("positions must be provided")
        if isinstance(positions, list):
            array = np.asarray(positions, dtype=float)
        elif isinstance(positions, np.ndarray):
            array = positions.astype(float)
        else:
            raise TypeError("positions must be list or numpy array")
        if array.shape != (self.N, 3):
            raise ValueError("positions must have shape (N, 3)")
        self.positions = array

    def _set_lattice(self, lattice):
        if lattice is None:
            raise ValueError("lattice must be provided")
        if isinstance(lattice, list):
            array = np.asarray(lattice, dtype=float)
        elif isinstance(lattice, np.ndarray):
            array = lattice.astype(float)
        else:
            raise TypeError("lattice must be list or numpy array")
        flat = array.reshape(-1)
        if flat.size != 9:
            raise ValueError("lattice must contain nine values")
        self.lattice = flat.reshape(3, 3)

    def _set_pbc(self, pbc):
        if pbc is None:
            self.pbc = (True, True, True)
            return
        if isinstance(pbc, (list, tuple)) and len(pbc) == 3:
            self.pbc = tuple(bool(x) for x in pbc)
            return
        if isinstance(pbc, np.ndarray):
            array = np.asarray(pbc, dtype=bool)
            if array.shape != (3,):
                raise ValueError("pbc must contain three values")
            self.pbc = tuple(array.tolist())
            return
        raise TypeError("pbc must be a list, tuple, or numpy array of three bool values")

    def _set_masses(self, masses):
        if masses is None:
            array = np.array([MASS_TABLE.get(species, 0.0) for species in self.species], dtype=float)
        elif isinstance(masses, list):
            array = np.asarray(masses, dtype=float)
        elif isinstance(masses, np.ndarray):
            array = masses.astype(float)
        else:
            raise TypeError("masses must be list or numpy array")
        if array.shape != (self.N,):
            raise ValueError("masses must have length N")
        self.masses = array

    def _set_velocities(self, velocities):
        if velocities is None:
            self.velocities = None
            return
        if isinstance(velocities, list):
            array = np.asarray(velocities, dtype=float)
        elif isinstance(velocities, np.ndarray):
            array = velocities.astype(float)
        else:
            raise TypeError("velocities must be list or numpy array")
        if array.shape != (self.N, 3):
            raise ValueError("velocities must have shape (N, 3)")
        self.velocities = array

    def _set_groups(self, groups):
        if groups is None:
            self.groups = None
            return
        parsed = []
        for group in groups:
            if isinstance(group, list):
                array = np.asarray(group, dtype=int)
            elif isinstance(group, np.ndarray):
                array = group.astype(int)
            else:
                raise TypeError("each group must be list or numpy array")
            if len(array) != self.N:
                raise ValueError("each group must match number of atoms")
            parsed.append(array)
        self.groups = parsed

    def set_info(self, info):
        if info is None:
            self.info = {}
        elif isinstance(info, dict):
            self.info = info
        else:
            raise TypeError("info must be a dictionary")

    def get_cell(self):
        return self.lattice.copy()

    def get_pbc(self):
        return tuple(self.pbc)

    def get_volume(self):
        cell = self.get_cell()
        return float(np.dot(np.cross(cell[0], cell[1]), cell[2]))

    def zero_momentum(self):
        """Shift velocities so total momentum matches `target` (defaults to zero)."""
        total_mass = np.sum(self.masses)
        current_momentum = np.sum(self.masses[:, None] * self.velocities, axis=0)
        drift_velocity = (current_momentum) / total_mass
        self.velocities = self.velocities - drift_velocity


def write_run(parameters: list[str]):
    """
    Write the input parameters to a gpumd file 'run.in'.
    """
    with open('run.in','w') as f:
        for i in parameters:
            f.write(i+'\n')

def dump_xyz(filename: str, atoms: Structure):
    def is_valid_key(key: str) -> bool:
        return key in atoms.info and atoms.info[key] is not None and all(v is not None for v in atoms.info[key])
    
    valid_keys = {key: is_valid_key(key) for key in ['energy','virial', 'forces']}
    
    with open(filename, 'a') as f:
        out_string = ""
        out_string += str(int(len(atoms))) + "\n"
        out_string += "pbc=\"" + " ".join(["T" if pbc_value else "F" for pbc_value in atoms.get_pbc()]) + "\" "
        out_string += "Lattice=\"" + " ".join(list(map(str, atoms.get_cell().reshape(-1)))) + "\" "
        if valid_keys['energy']:
            out_string += " energy=" + str(atoms.info['energy']) + " "
        if valid_keys['virial']:
            out_string += "virial=\"" + " ".join(list(map(str, atoms.info['virial']))) + "\" "
        out_string += "Properties=species:S:1:pos:R:3:mass:R:1"
        if atoms.velocities is not None:
            out_string += ":vel:R:3"
        if valid_keys['forces']:
            out_string += ":force:R:3"
        if atoms.groups is not None:
            out_string += f":group:I:{len(atoms.groups)}"
        out_string += "\n"
        for atom in atoms:
            out_string += '{:2} {:>15.8e} {:>15.8e} {:>15.8e} {:>15.8e}'.format(atom.species, *atom.position, atom.mass)
            if atoms.velocities is not None:
                out_string += ' {:>15.8e} {:>15.8e} {:>15.8e}'.format(*atoms.velocities[atom.index])
            if valid_keys['forces']:
                out_string += ' {:>15.8e} {:>15.8e} {:>15.8e}'.format(*atoms.info['forces'][atom.index])
            if atoms.groups is not None:
                for group in atoms.groups:
                    out_string += f" {int(group[atom.index])}"
            out_string += '\n'
        f.write(out_string)

def _parsed_properties(comment: str) -> dict[str, slice]:
    properties_str = comment.split('properties=')[1].split()[0]
    properties = properties_str.split(':')
    parsed_properties = {}
    start = 0
    for i in range(0, len(properties), 3):
        property_name = properties[i]
        property_count = int(properties[i+2])
        parsed_properties[property_name] = slice(start, start + property_count)
        start += property_count
    return parsed_properties

def _normalize_optional(seq: list):
    if not seq or None in seq:
        return None
    return seq

def _read_species(words_in_line: list[str], parsed_properties: dict[str, slice]) -> str:
    species_slice = parsed_properties['species']
    species = words_in_line[species_slice]
    species = species[0].lower().capitalize()
    return species

def _read_positions(words_in_line: list[str], parsed_properties: dict[str, slice]) -> tuple[float, float, float]:
    pos_slice = parsed_properties['pos']
    pos = words_in_line[pos_slice]
    return tuple(float(p) for p in pos)

def _read_mass(words_in_line: list[str], parsed_properties: dict[str, slice]) -> float | None:
    if 'mass' in parsed_properties:
        mass_slice = parsed_properties['mass']
        mass = words_in_line[mass_slice]
        return float(mass[0])
    else:
        return None

def _read_force(words_in_line: list[str], parsed_properties: dict[str, slice]) -> tuple[float, float, float] | None:
    force_key = 'forces' if 'forces' in parsed_properties else 'force'
    if force_key in parsed_properties:
        force_slice = parsed_properties[force_key]
        force = words_in_line[force_slice]
        return tuple(float(f) for f in force)
    else:
        return None
    
def _read_velocity(words_in_line: list[str], parsed_properties: dict[str, slice]) -> tuple[float, float, float] | None:
    if 'vel' in parsed_properties:
        vel_slice = parsed_properties['vel']
        vel = words_in_line[vel_slice]
        return tuple(float(v) for v in vel)
    return None

def _read_group(words_in_line: list[str], parsed_properties: dict[str, slice]) -> int | None:
    if 'group' in parsed_properties:
        group_slice = parsed_properties['group']
        group = words_in_line[group_slice]
        return tuple(int(g) for g in group)
    else:
        return None

def read_xyz(filename: str) -> Structure:
    """
    Read the atomic positions and other information from a file in XYZ format.
    """
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            species = []
            positions = []
            masses = []
            forces = []
            velocities = []
            group = []
            natoms = int(line.strip())
            comment = f.readline().lower().strip()
            if "pbc=\"" in comment:
                pbc_str = comment.split("pbc=\"")[1].split("\"")[0].strip()
                pbc = [True if pbc_value == "t" else False for pbc_value in pbc_str.split()]
            else:
                pbc = [True, True, True]
            lattice_str = comment.split("lattice=\"")[1].split("\"")[0].strip()
            lattice = [list(map(float, row.split())) for row in lattice_str.split(" ")]
            if "energy=" in comment:
                energy = float(comment.split("energy=")[1].split()[0])
            else: 
                energy = None
            if "virial=" in comment:
                virial = comment.split("virial=\"")[1].split("\"")[0].strip()
                virial = np.array([float(x) for x in virial.split()])
            else:
                virial = None
            parsed_properties_dict = _parsed_properties(comment)
            for _ in range(natoms):
                line = f.readline()
                words_in_line = line.split()
                species.append(_read_species(words_in_line, parsed_properties_dict))
                positions.append(_read_positions(words_in_line, parsed_properties_dict))
                masses.append(_read_mass(words_in_line, parsed_properties_dict))
                forces.append(_read_force(words_in_line, parsed_properties_dict))
                velocities.append(_read_velocity(words_in_line, parsed_properties_dict))
                group.append(_read_group(words_in_line, parsed_properties_dict))
            velocities = _normalize_optional(velocities)
            group = _normalize_optional(group)
            if group is not None:
                # transpose from per-atom tuples to per-group arrays
                group_arrays = list(zip(*group))
                groups = [np.asarray(column, dtype=int) for column in group_arrays]
            else:
                groups = None
            forces = _normalize_optional(forces)
            masses = _normalize_optional(masses)
            atoms = Structure(
                species=species,
                positions=positions,
                lattice=lattice,
                pbc=pbc,
                masses=masses,
                velocities=velocities,
                groups=groups,
                info={'energy': energy, 'virial': virial, 'forces': forces})
    return atoms

def gpumd(dirname = None, atoms = None, run_in = None, nep_path = None, gpumd_path = 'gpumd'):
    if os.path.exists(dirname):
        raise FileExistsError('Directory already exists')
    os.makedirs(dirname)
    if os.path.exists(nep_path):
        shutil.copy(nep_path, dirname)
    else:
        raise FileNotFoundError('nep.txt does not exist')
    original_directory = os.getcwd()
    os.chdir(dirname)
    write_run(run_in)
    dump_xyz('model.xyz', atoms)
    os.system(gpumd_path)

    os.chdir(original_directory)
