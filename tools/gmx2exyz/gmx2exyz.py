#Convert the trr trajectory of gmx to the exyz trajectory
#write in the potential energy (eV) and atomic forces (eV/A)
#Before use, run the gmx energy command to obtain the energy. xvg file that records potential energy

#Author:Zherui Chen (chenzherui0124@foxmail.com)

import MDAnalysis as mda

u = mda.Universe('nvt.tpr', 'test.trr')

sel = u.select_atoms('all', updating=False)

# Load the energy file
with open('energy.xvg', 'r') as f:
    for i in range(24):
        f.readline()  # Skip the first 24 lines of comments
    energies = [float(line.split()[1]) / 96.48533212331 for line in f]  # Convert from kJ/mol to eV

with open('your_exyz_file.xyz', 'w') as f:
    for i, ts in enumerate(u.trajectory):

        f.write('{}\n'.format(sel.n_atoms))

        box = ts.dimensions[:3]
        f.write('energy={:.8f} config_type=gmx2xyz pbc="T T T" Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}" Properties=species:S:1:pos:R:3:force:R:3\n'.format(energies[i], box[0], box[1], box[2]))

        for atom in sel:
            # Get the force on the atom
            force = atom.force / 96.48533212331 # Convert from kJ/(mol*A) to eV/A

            # Write the atom element, position, and force to the output file
            f.write('{} {} {} {} {} {} {}\n'.format(atom.element, atom.position[0], atom.position[1], atom.position[2], force[0], force[1], force[2]))




