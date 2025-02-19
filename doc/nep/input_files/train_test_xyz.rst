.. _train_test_xyz:
.. index::
   single: train.xyz (input file)
   single: test.xyz (input file)

``train.xyz`` and ``test.xyz``
==============================

The :attr:`train.xyz` file, which contains the training data for the construction of a :term:`NEP` model, and the :attr:`test.xyz` file, which contains the corresponding test data, both need to be provided in `extended xyz file format <https://github.com/libAtoms/extxyz>`_.
Each structure (or configuration or frame) occupies :math:`N+2` lines, where :math:`N` is the number of atoms in the structure.

Format for a single structure
-----------------------------

Line 1
^^^^^^
The first line should only contain one field, which is the number of atoms in the structure :math:`N`.

Line 2
^^^^^^

This line consists of a number of ``keyword=value`` pairs separated by spaces.
Spaces before and after ``=`` are allowed.
All the characters are case-insensitive.
``value`` can be a single item or a number of items enclosed by double quotes, such as ``keyword="value_1 value_2 value_3"``.
Here, the different values are separated by spaces and spaces after the left ``"`` and before the right ``"`` are allowed.
For example, one can write ``keyword=" value_1 value_2 value_3 "``.

Essentially any keyword is allowed, but we only read the following ones:

* :attr:`lattice="ax ay az bx by bz cx cy cz"` is mandatory and gives the cell vectors:

  .. math::
     
     \boldsymbol{a} &= a_x \boldsymbol{e}_x + a_y \boldsymbol{e}_y + a_z \boldsymbol{e}_z \\
     \boldsymbol{b} &= b_x \boldsymbol{e}_x + b_y \boldsymbol{e}_y + b_z \boldsymbol{e}_z \\
     \boldsymbol{c} &= c_x \boldsymbol{e}_x + c_y \boldsymbol{e}_y + c_z \boldsymbol{e}_z

* :attr:`energy=energy_value` such as :attr:`energy=-123.4` is mandatory and gives the target energy of the structure, which is :math:`-123.4` eV in this example.
* :attr:`virial="vxx vxy vxz vyx vyy vyz vzx vzy vzz"` is optional and gives the :math:`3\times3` virial tensor of the structure in eV.
  Positive and negative values represent compressed and stretched states, respectively.
* :attr:`stress="sxx sxy sxz syx syy syz szx szy szz"` is optional and gives the :math:`3\times3` stress tensor of the structure in eV/Å\ :math:`^3`.
  Positive and negative values represent stretched and compressed states, respectively.
  If both :attr:`virial` and :attr:`stress` are present the former is used.
* :attr:`weight=relative_weight` is optional and gives the relative weight for the current structure in the total loss function.
* :attr:`properties=property_name:data_type:number_of_columns` is mandatory but only read the following items:
  
  * :attr:`species:S:1` chemical symbol in the periodic table (case-sensitive)
  * :attr:`pos:R:3` position vector
  * :attr:`force:R:3` or :attr:`forces:R:3` target force vector

* If a dipole model is to be trained, energy, virial, stress, and force will be ignored and one should additionally provide :attr:`dipole="dx dy dz"`, which is the dipole vector of the structure. 

* If a polarizability model is to be trained, energy, virial, stress, force, and dipole will be ignored and one should additionally provide :attr:`pol="pxx pxy pxz pyx pyy pyz pzx pzy pzz"`, which is the polarizability tensor of the structure.

Starting from line 3
^^^^^^^^^^^^^^^^^^^^

Each line should contain the same number of items, which are determined by the :attr:`property` keywords on line 2.

Units
-----
* Length and position are expected in units of Ångstrom.
* The energy is expected  in units of eV.
* Forces are expected in units of eV/Å.
* Virials are expected in units of eV (such that the virial divided by the volume yields the stress).
* Dipole and polarizability can be in arbitrary units (such as the Hartree atomic units) as liked (and remembered) by the user.

Tips
----
* Periodic boundary conditions are always assumed for all directions in each configuration.
  When the box thickness in a direction is smaller than twice of the radial cutoff distance, the code will internally replicate the box in that direction.
* The minimal number of atoms in a configuration is 1.
  The user is responsible for choosing a sensible reference energy when preparing the energy data.
  But this is not crucial as the absolute energies are not relevant in the present context.
  However, because NEP training uses single precision, accuracy will be lost if any reference energy is smaller than -100 eV/atom. The code will give a warning message in this case.
* The energy and virial data refer to the total energy and virial for the system.
  They are not per-atom but per-cell quantities.
