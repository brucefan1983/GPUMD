#
# Purpose:
#     Merge the box information, atomic coordinates,
#     and atomic forces outputted by CP2K into the xyz file
# Run:
#     python cp2k2xyz.py ${cp2k}  # cp2k is a folder path.
#

import sys
import os

def Get_CP2K_Filename(folder):

    pos_file = None
    frc_file = None
    cell_file = None

    for fn in os.listdir(folder):
        if ".xyz" in fn:
            if "-pos-" in fn:
                pos_file = fn
            elif "-frc-" in fn:
                frc_file = fn

        elif ".cell" in fn:
            cell_file = fn

    if pos_file == None or frc_file == None or cell_file == None:
        print(f"Please check your file in {folder}")
        raise "Errors with files."
    else:
        return pos_file, frc_file, cell_file


def CP2K2XYZ(folder, fnames=None, output_file=None):

    if fnames == None :
        pos_file, frc_file, cell_file = Get_CP2K_Filename(folder)
    else:
        pos_file = fnames[0]
        frc_file = fnames[1]
        cell_file = fnames[2]

    os.makedirs("CP2K2XYZ", exist_ok=True)
    if output_file == None:
        output_file = "merged_cp2k.xyz"

    pos_file = os.path.join(folder, pos_file)
    frc_file = os.path.join(folder, frc_file)
    cell_file = os.path.join(folder, cell_file)
    output_file = os.path.join("CP2K2XYZ", output_file)
    print("Input:",pos_file,frc_file,cell_file)
    print("Output:",output_file)

    with open(pos_file, 'r') as pf, open(frc_file, 'r') as ff, \
         open(cell_file, 'r') as cf, open(output_file, 'w') as of:
        cf.readline()  # Skip the header line in the cell_file

        while True:
            pos_header = pf.readline()
            frc_header = ff.readline()

            if not pos_header or not frc_header:
                break

            num_atoms = int(pos_header.strip().split()[0])
            of.write(f"{num_atoms}\n")
            frc_info_line = ff.readline()
            pf.readline()  # Skip the corresponding line in the pos_file

            energy = float(frc_info_line.strip().split("E =")[-1]) * 27.211386245988

            # Read and process the cell information
            cell_line = cf.readline().strip().split()
            # Only read the Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz columns
            lattice = " ".join(cell_line[2:11])

            of.write(f"energy={energy:.10f} config_type=cp2k2xyz pbc=\"T T T\" ")
            of.write(f"Lattice=\"{lattice}\" Properties=species:S:1:pos:R:3:force:R:3\n")

            for _ in range(num_atoms):
                pos_line = pf.readline().strip().split()
                frc_line = ff.readline().strip().split()

                if len(pos_line) < 4 or len(frc_line) < 4:
                    break

                force_x = float(frc_line[1]) * 51.42206747632590000
                force_y = float(frc_line[2]) * 51.42206747632590000
                force_z = float(frc_line[3]) * 51.42206747632590000

                of.write(f"{pos_line[0]} {pos_line[1]} {pos_line[2]} {pos_line[3]} ")
                of.write(f"{force_x:.10f} {force_y:.10f} {force_z:.10f}\n")

                
if __name__ == "__main__":

    dirpath = sys.argv[1]
    CP2K2XYZ(dirpath)
