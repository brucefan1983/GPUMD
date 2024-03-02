.. _model_xyz:
.. index::
   single: gpumd input files; Simulation model
   single: gpumd input files; Atomic configuration
   single: model.xyz

Simulation model (``model.xyz``)
================================

The ``model.xyz`` file defines the simulation model, including for example the initial positions, velocities, and boundary conditions.
The file needs to be provided in extended XYZ format, which is described `here <https://github.com/libAtoms/extxyz>`_.

File format
-----------

Line 1
^^^^^^

The first line should have one item only, which is the number of atoms in the model :math:`N`.

Line 2
^^^^^^

This line consists of a number of ``keyword=value`` pairs separated by spaces.
Spaces before and after ``=`` *are allowed*.
All the characters are case-insensitive.
``value`` can be a single item or a number of items enclosed by double quotes, such as ``keyword="value_1 value_2 value_3"``.
Here, the different values are separated by spaces and spaces after the left ``"`` and before the right ``"`` are allowed.
For example, one can write ``keyword=" value_1 value_2 value_3 "``.

Essentially any keyword is allowed, but we only read the following ones:

* (*Optional*) ``pbc="pbc_a pbc_b pbc_c"``, where ``pbc_a``, ``pbc_b``, and ``pbc_c`` can be ``T`` or ``F``, which means box is periodic or non-periodic (free, or open) in the :math:`\boldsymbol{a}`, :math:`\boldsymbol{b}`, and :math:`\boldsymbol{c}` directions.
  We use the minimum-image convention to account for the periodic boundary conditions for non-:term:`NEP` potentials but will consider sufficiently many periodic images for the :term:`NEP` potentials.
  Therefore, one needs to make sure that the box size is large enough (the thickness in each direction is larger than twice of the potential cutoff) to incorporate enough neighbors for non-:term:`NEP` potentials but does not need to care about this for NEP potentials.
  The default is ``pbc="T T T"``
* (*Mandatory*) ``lattice="ax ay az bx by bz cx cy cz"`` specifies the cell vectors:

  .. math::

     \boldsymbol{a} = a_x \boldsymbol{e}_x + a_y \boldsymbol{e}_y + a_z \boldsymbol{e}_z\\
     \boldsymbol{b} = b_x \boldsymbol{e}_x + b_y \boldsymbol{e}_y + b_z \boldsymbol{e}_z\\
     \boldsymbol{c} = c_x \boldsymbol{e}_x + c_y \boldsymbol{e}_y + c_z \boldsymbol{e}_z

* ``properties=property_name:data_type:number_of_columns``

  We only read the following items:

  * ``species:S:1`` atom type (*Mandatory*)
  * ``pos:R:3`` position vector (*Mandatory*)
  * ``mass:R:1`` mass (*Optional*: default mass values will be used when this is missing) 
  * ``vel:R:3`` velocity vector (*Optional*)
  * ``group:I:number_of_grouping_methods`` grouping methods (*Optional*)


Line 3 and forward
^^^^^^^^^^^^^^^^^^

Each line must contain the same number of items, which are determined by the ``property`` keyword on line 2.
The meaning of the grouping methods is illustrated by the example below.

Units
-----

The mass should be given in units of the unified atomic mass unit (amu). 
The cell dimensions and atom coordinates should be given in units of Ångstrom. 
Velocities need to be specified in units of Å/fs.

Example
-------

Assume we have the following ``model.xyz`` file:

.. code::

   10
   pbc="T F F" lattice="4 0 0 0 1 0 0 0 1" properties=species:S:1:pos:R:3:group:I:3
   C  0 0 0 0 0 0
   Si 1 0 0 0 1 0
   C  2 0 0 0 2 0
   Si 3 0 0 0 3 0
   C  4 0 0 0 4 0
   Si 5 0 0 1 5 0
   C  6 0 0 1 6 0
   Si 7 0 0 1 7 0
   C  8 0 0 1 8 0
   Si 9 0 0 1 9 0

This means

* There are 10 atoms.
* Use periodic boundary conditions in the :math:`x` direction and free boundary conditions in the other directions.
* The box lengths in the three directions are respectively 4 Å, 1 Å, and 1 Å.
* There are 5 carbon atoms and 5 silicon atoms. The default masses will be used for the atoms (as there are no specific values given in the input file).
* The 10 atoms are located along a line in the :math:`x` direction with equal spacing, from 0 Å to 9 Å. 
* There is no velocity data in this file.
* There are 3 grouping methods

  * In grouping method 0, atom 0 to atom 4 have group label 0 (which means they are in group 0), and atom 5 to atom 9 have group label 1 (which means they are in group 1).
  * In grouping method 1, atom :math:`m` (:math:`0\leq m \leq 9`) has group label :math:`m`. That is, each group consists of a single atom.
  * In grouping method 2, all the atoms have group label 0. That is, all the atoms are in the same group.
