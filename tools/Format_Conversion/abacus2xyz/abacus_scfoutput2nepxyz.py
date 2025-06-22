import os, sys, json
import numpy as np
from ase.io import read, write

def main():
    if len(sys.argv) != 3:
        print("Usage: python single-abacus2nep.py <dir> <xyz>")
        sys.exit(1)
if __name__ == "__main__":
    main()

def get_scf_info(root):
    scf_nmax = None
    input_file = os.path.join(root, "INPUT")
    if os.path.exists(input_file):
        with open(input_file, 'r') as file:
            for line in file:
                line = line.strip()
                if 'scf_nmax' in line:
                    scf_nmax = int(line.split()[1])
    else:
        scf_nmax = 100
        print(f"{root} doesn't know the scf_max value, default to 100")
    return scf_nmax

for root, dirs, files in os.walk(os.path.abspath(sys.argv[1])):
    scf_count, Total_Time, virial = 0, 0, None
    if "running_scf.log" in files:
        log_file = os.path.join(root, "running_scf.log")
        scf_nmax = get_scf_info(root)
        with open(log_file, 'r') as file_log:
            for line in file_log:
                line = line.strip()
                scf_count += line.count("ALGORITHM")
                Total_Time += line.count("Total  Time")
        if scf_count == scf_nmax or Total_Time == 0:
            print(f"Directory {root} has not completed the calculation or has not converged")
            continue
        atoms = read(log_file, format='abacus-out') #pip install git+https://gitlab.com/1041176461/ase-abacus.git
        natoms = len(atoms)
        cell = np.concatenate([atoms.get_cell()[0], atoms.get_cell()[1], atoms.get_cell()[2]])
        energy = atoms.get_potential_energy()
        if atoms.calc.get_stress() is not None:
            xx,yy,zz,yz,xz,xy = atoms.calc.get_stress()
            stresses = [xx, xy, xz, xy, yy, yz, xz, yz, zz]
            virial = [f"{-s:.10f}" for s in [stress * atoms.get_volume() for stress in stresses]]
        else:
            print(f"This structure does not calculate stress in {root}, please add cal_stress in INPUT")
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        forces = atoms.get_forces()
    elif "abacus.json" in files and "running_scf.log" not in files:
        print(f"Directory {root} with abacus.json")
        json_file = os.path.join(root, "abacus.json")
        scf_nmax = get_scf_info(root)
        with open(json_file, 'r', encoding='utf-8') as file_json:
            data = json.load(file_json)
        scf_count = len(data['output'][0]['scf'])
        if scf_count == scf_nmax:
            print(f"Directory {root} has not completed the calculation or has not converged")
            continue
        natoms = data['init']['natom']
        energy = data['output'][0]['energy']
        cells = data['output'][0]['cell']
        cell = np.concatenate([cells[0], cells[1], cells[2]])
        symbols = data['init']['label']
        positions = data['output'][0]['coordinate']
        forces = data['output'][0]['force']
        if 'stress' in open(json_file, 'r').read():
            volume = np.abs(np.dot(cells[0], np.cross(cells[1], cells[2])))
            stresses = data['output'][0]['stress']
            stress = np.concatenate([stresses[0], stresses[1], stresses[2]])
            virial = [s * (volume / 1602.1766208) for s in stress]
        else:
            print(f"This structure does not calculate stress in {root}, please add cal_stress in INPUT")
    elif "devie.log" in files and"abacus.json" not in files and "running_scf.log" not in files:
        print(f"{root} does not have the required output files, please check")
    else:
        continue

    with open(sys.argv[2], 'w') as f:
        f.write(f"{natoms}\n")
        lattice_str = " ".join(map(str, cell))
        if virial is not None:
            virial_str = " ".join(map(str, virial))
            f.write(f"energy={energy} Lattice=\"{lattice_str}\" Virial=\"{virial_str}\" config_type={root} Properties=species:S:1:pos:R:3:forces:R:3 \n")
        else:
            f.write(f"energy={energy} Lattice=\"{lattice_str}\" config_type={root} Properties=species:S:1:pos:R:3:forces:R:3 \n")
        for i in range(natoms):
            f.write(f"{symbols[i]:<20}{positions[i][0]:20.10f}{positions[i][1]:20.10f}{positions[i][2]:20.10f}{forces[i][0]:20.10f}{forces[i][1]:20.10f}{forces[i][2]:20.10f} \n")  

'''
#需要保留到小数点更多位请使用上面代码, 仅针对log文件
os.system("if [ -f sys.argv[2] ]; then rm aimd.xyz;fi")
for root, dirs, files in os.walk(os.path.abspath(sys.argv[1])):
    if "running_scf.log" in files:
        scf_count, Total_Time = 0, 0
        log_file = os.path.join(root, "running_scf.log")
        scf_nmax = get_scf_info(root)
        with open(log_file, 'r') as file_log:
            for line in file_log:
                line = line.strip()
                scf_count += line.count("ALGORITHM")
                Total_Time += line.count("Total  Time")
        if scf_count == scf_nmax or Total_Time == 0:
            print(f"Directory {root} has not completed the calculation or has not converged")
            continue
        atoms = read(log_file, format='abacus-out') #pip install git+https://gitlab.com/1041176461/ase-abacus.git
        atoms.set_momenta(None)
        atoms.set_initial_magnetic_moments(None)
        if 'TOTAL-STRESS' in open(log_file, 'r').read():
            xx,yy,zz,yz,xz,xy = -atoms.calc.get_stress()*atoms.get_volume()
            atoms.info['virial']= np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
            del atoms.calc.results['stress']
        else:
            print(f"This structure does not calculate stress in {root}, please add cal_stress in INPUT")
        atoms.info['config_type'] = root
    write(sys.argv[2], atoms, format='extxyz', append=True)
'''

