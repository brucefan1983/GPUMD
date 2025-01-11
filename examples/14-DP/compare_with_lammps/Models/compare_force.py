from ase.io import read, write
import numpy as np
import sys


def compare_natoms(gpumd, lammps):
    """
    Compare the number of atoms between GPUMD and LAMMPS structures.
    """
    ngmd = gpumd.get_global_number_of_atoms()
    nlmp = lammps.get_global_number_of_atoms()
    if ngmd == nlmp:
        print(f" >> There are the same number of atoms: {ngmd}")
        return ngmd
    else:
        print(f"GPUMD: {ngmd}, LAMMPS: {nlmp}")
        raise ValueError("Mismatch in atom counts between GPUMD and LAMMPS.")

def compare_ntypes(gpumd, lammps, lammps_type_map=None):
    """
    Compare the types of atoms between GPUMD and LAMMPS structures.
    """
    gpumd_symbols = gpumd.get_chemical_symbols()
    if lammps_type_map == None:
        lammps_symbols = lammps.get_chemical_symbols()
    else:
        nlmp_atomic = lammps.get_atomic_numbers()
        lammps_symbols = [lammps_type_map[i-1] for i in nlmp_atomic]

    if gpumd_symbols == lammps_symbols:
        print(f" >> There are the same types of atoms: {gpumd_symbols[:5]} (showing first 5).")
        return gpumd_symbols
    else:
        print(f"GPUMD (first 5 types): {gpumd_symbols[:5]}, LAMMPS (first 5 types): {lammps_symbols[:5]}")
        # raise ValueError("Mismatch in atom types between GPUMD and LAMMPS.")


def compare_positions(gpumd, lammps):
    """
    Compare atomic positions between GPUMD and LAMMPS structures.
    """
    gpumd_positions = gpumd.get_positions()
    lammps_positions = lammps.get_positions()

    if np.allclose(gpumd_positions, lammps_positions, atol=1e-5):
        print(" >> Atomic positions are nearly identical.")
        return False
    else:
        print(" >> Atomic positions differ.")
        # raise ValueError("Mismatch in atom positions between GPUMD and LAMMPS.")


def compare_forces(gpumd, lammps):
    """
    Compare atomic forces between GPUMD and LAMMPS structures.
    """
    gpumd_forces = gpumd.get_forces()
    lammps_forces = lammps.get_forces()

    if np.allclose(gpumd_forces, lammps_forces, atol=1e-5):
        print(" >> Atomic forces are nearly identical.")
        return False
    else:
        ndiff = np.sum(np.abs(gpumd_forces-lammps_forces)>1e-5)
        print(f" >> Atomic forces differ {ndiff}.")
        return np.abs(gpumd_forces - lammps_forces)


def compare_energies(gpumd, lammps):
    """
    Compare atomic energies between GPUMD and LAMMPS structures.
    """
    gpumd_energies = gpumd.get_array("energy_atom")
    lammps_energies = lammps.get_array("c_pe").reshape(-1)

    if np.allclose(gpumd_energies, lammps_energies, atol=1e-5):
        print(" >> Atomic energies are nearly identical.")
        return False
    else:
        ndiff = np.sum(np.abs(gpumd_energies-lammps_energies)>1e-5)
        print(f" >> Atomic energies differ {ndiff}.")
        return np.abs(gpumd_energies - lammps_energies)


def compare_virials(fgpumd, lammps):
    """
    Compare atomic energies between GPUMD and LAMMPS structures.
    """
    joul_to_eV = 6.241506 * 1e-7
    scale = joul_to_eV * -1
    gpumd_v3 = np.loadtxt(fgpumd+".out")
    # gpumd_v3 = np.loadtxt(fgpumd+"-v3.out")
    # gpumd_v6 = np.loadtxt(fgpumd+"-v6.out")
    # gpumd_v9 = np.loadtxt(fgpumd+"-v9.out")
    #print(gpumd[0],gpumd[192],gpumd[192*2])
    gpumd_virials_3 = gpumd_v3.reshape((-1,3), order='F')
    # gpumd_virials_6 = gpumd_v6.reshape((-1,3), order='F')
    # gpumd_virials_9 = gpumd_v9.reshape((-1,3), order='F')
    #print(gpumd_virials[:3,:])
    lmp_vxx = lammps.get_array("v_xx").reshape(-1)
    lmp_vyy = lammps.get_array("v_yy").reshape(-1)
    lmp_vzz = lammps.get_array("v_zz").reshape(-1)
    lammps_virials_3 = np.array([lmp_vxx,lmp_vyy,lmp_vzz]).T * scale
    lmp_vxy = lammps.get_array("v_xy").reshape(-1)
    lmp_vxz = lammps.get_array("v_xz").reshape(-1)
    lmp_vyz = lammps.get_array("v_yz").reshape(-1)
    lammps_virials_6 = np.array([lmp_vxy,lmp_vxz,lmp_vyz]).T * scale
    lmp_vyx = lammps.get_array("v_yx").reshape(-1)
    lmp_vzx = lammps.get_array("v_zx").reshape(-1)
    lmp_vzy = lammps.get_array("v_zy").reshape(-1)
    lammps_virials_9 = np.array([lmp_vyx,lmp_vzx,lmp_vzy]).T * scale

    if np.allclose(gpumd_virials_3, lammps_virials_3, atol=1e-5):
        print(" >> Atomic virials are nearly identical. _v3")
        # return False
    else:
        ndiff = np.sum(np.abs(gpumd_virials_3-lammps_virials_3)>1e-5)
        print(f" >> Atomic virials differ {ndiff}. _v3")
        # return np.abs(gpumd_virials_3 - lammps_virials_3)
    return 0

    if np.allclose(gpumd_virials_6, lammps_virials_6, atol=1e-5):
        print(" >> Atomic virials are nearly identical. _v6")
        # return False
    else:
        ndiff = np.sum(np.abs(gpumd_virials_6-lammps_virials_6)>1e-5)
        print(f" >> Atomic virials differ {ndiff}. _v6")
        # return np.abs(gpumd_virials_6 - lammps_virials_6)

    if np.allclose(gpumd_virials_9, lammps_virials_9, atol=1e-5):
        print(" >> Atomic virials are nearly identical. _v9")
        # return False
    else:
        ndiff = np.sum(np.abs(gpumd_virials_9-lammps_virials_9)>1e-5)
        print(f" >> Atomic virials differ {ndiff}. _v9")
        # return np.abs(gpumd_virials_9 - lammps_virials_9)


def save_differences_to_extxyz(gpumd, diff_forces, diff_energies, output_file="differences.xyz"):
    """
    Save the differences in positions, forces, and energies to a new extxyz file.
    """
    has_force_diff = isinstance(diff_forces, np.ndarray) and diff_forces.any()
    has_energy_diff = isinstance(diff_energies, np.ndarray) and diff_energies.any()
    if has_force_diff:
        gpumd.new_array("diff_forces", diff_forces)
    if has_energy_diff:
        gpumd.new_array("diff_energies", diff_energies)
    if has_force_diff or has_energy_diff:
        print(f" >> Differences saved to {output_file}")
        write(output_file, gpumd)
    #else:
    #    print(f"Only save the structure information {output_file}")


in_str = sys.argv[1]
out_name = f"{in_str}-h2o"

gpumd_xyz = read(f"gmd-wat-{in_str}/dump.xyz", format="extxyz", index="0")
lammps_strj = read(f"lmp-wat-{in_str}/dump.lammpstrj", format="lammps-dump-text", index="0")

natoms = compare_natoms(gpumd_xyz, lammps_strj)

# types = compare_ntypes(gpumd_xyz, lammps_strj, lammps_type_map=['O', 'H'])
types = compare_ntypes(gpumd_xyz, lammps_strj, lammps_type_map=None)

diff_positions = compare_positions(gpumd_xyz, lammps_strj)
diff_forces = compare_forces(gpumd_xyz, lammps_strj)
diff_energies = compare_energies(gpumd_xyz, lammps_strj)
diff_virials = compare_virials(f"gmd-wat-{in_str}/compute", lammps_strj)

save_differences_to_extxyz(lammps_strj, diff_forces, diff_energies, output_file=f"diff-{out_name}.xyz")

