.. _active_xyz:
.. index::
   single: active.xyz (output file)

``active.xyz``
================

File containing atomistic positions, velocities, forces and uncertainties for structures written during on-the-fly active learning.
It is generated when invoking the :ref:`active <kw_active>` keyword.
A structure is written when its uncertainty exceeds the threshold :math:`\delta` or is NaN.
If no structure is written during a run, the run adds nothing to this file.

File format
-----------
This file is in the `extended XYZ format <https://github.com/libAtoms/extxyz>`_.
The second line of a frame holds the time `Time` in fs, the uncertainty :math:`\sigma_f` as `uncertainty` in eV/Å, the periodic boundary conditions `pbc`, the cell `Lattice` in Å, the energy `energy` and the virial `virial` in eV, and the pressure tensor `stress` in eV/Å³.
The energy, virial and pressure tensor are those of the main potential, and the pressure tensor includes the kinetic contribution of the velocities.
Each atom line holds the species, the position in Å, and, as selected by the keyword, the velocity in Å/fs, the force in eV/Å and the uncertainty of the atom in eV/Å.
The output mode for this file is append.
