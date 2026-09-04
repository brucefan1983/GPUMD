.. _kw_deposit:
.. index::
   single: deposit (keyword in run.in)

:attr:`deposit`
===============

This keyword is used to simulate a deposition process, where new atoms are periodically added to the system during a run.

Syntax
------

This keyword is used in one of the following two ways::

  deposit <interval> <direction> <height_min> [height_max] atom <element_1> <num_1> <velocity_1> [mass_1] [<element_2> <num_2> <velocity_2> [mass_2] ...] # usage 1
  deposit <interval> <direction> <height_min> [height_max] file <add_atom_file> [velocity]                                                          # usage 2

* :attr:`interval` is the deposition interval (number of steps) and must be a positive integer.
  The number of steps in the :ref:`run keyword <kw_run>` must be divisible by it.
* :attr:`direction` is the deposition direction: :attr:`0`, :attr:`1`, and :attr:`2` correspond to the x, y, and z directions, respectively.
  New atoms are placed at the given height along this direction and are given an initial velocity along it.
  The special value :attr:`-1` is only allowed in the second usage and means that the atoms are added at the exact positions and with the exact velocities given in the file.
* :attr:`height_min` and the optional :attr:`height_max` define the deposition height (in units of Ångstrom) along the deposition direction.
  If :attr:`height_max` is given, the height of each deposited atom is uniformly sampled between :attr:`height_min` and :attr:`height_max`; otherwise, all the atoms are deposited at :attr:`height_min`.
  When :attr:`direction` is :attr:`-1`, the height value(s) are not used.
* In the first usage, one or more triplets :attr:`<element> <num> <velocity>` are given, each optionally followed by a :attr:`mass`.
  For each triplet, :attr:`num` atoms of element :attr:`element` are added at each deposition, at random lateral positions in the simulation box, with an initial velocity :attr:`velocity` along the deposition direction.
  The deposited species must be specified by its element symbol, and the velocity is in units of Å/fs.
  The optional :attr:`mass` (in unified atomic mass units) is only meaningful when the :ref:`simulation model file <model_xyz>` has a mass property; if omitted, the standard mass of the element is used.
  It is ignored when the model file has no mass property.
* In the second usage, the atoms to be added at each deposition are read from the file :attr:`add_atom_file`, which contains one atom per row with 7 columns plus optional mass column::

    element x y z vx vy vz [mass]

  The first column must contain the element symbol of the deposited atom.
  The positions are in units of Ångstrom and the velocities are in units of Å/fs.
  When the :ref:`simulation model file <model_xyz>` has a mass property, an optional eighth column can be used to specify the mass of each atom; if omitted, the standard mass of the element is used.
  The eighth column is ignored when the model file has no mass property.
  With a non-negative :attr:`direction`, the coordinate along the deposition direction is replaced by the sampled height, and the optional :attr:`velocity` (in units of Å/fs) after the file name sets the velocity component along the deposition direction.
  With :attr:`direction` set to :attr:`-1`, no :attr:`velocity` is allowed after the file name and the file is used as it is.

At each deposition, the new atoms are appended to the model and the :ref:`run <kw_run>` is effectively split into sub-runs of :attr:`interval` steps, with one deposition between two consecutive sub-runs.
After each sub-run, the current structure is written to a file named :attr:`deposited_N.xyz` in extended XYZ format, including the velocities (and the mass if the model has a mass property, and the group labels if the model has grouping).
If the :ref:`simulation model file <model_xyz>` contains no velocity data, one extra sub-run of the pristine model is performed before the first deposition.

Example 1
---------

Deposit 10 H atoms every 10000 steps, at a height of 30 Å in the z direction, with an initial velocity of -0.1 Å/fs (towards decreasing z)::

   deposit 10000 2 30 atom H 10 -0.1

Example 2
---------

Deposit 10 H atoms with a mass of 2 for every 10000 steps, at a height of 30 Å in the z direction, with an initial velocity of -0.1 Å/fs (towards decreasing z)::

   deposit 10000 2 30 atom H 10 -0.1 2

Example 3
---------

Deposit 5 H atoms and 5 C atoms every 5000 steps, at heights uniformly sampled between 30 Å and 40 Å in the z direction::

   deposit 5000 2 30 40 atom H 5 -0.1 C 5 -0.1

Example 4
---------

Deposit atoms read from the file :attr:`add_atoms.txt` every 10000 steps, resetting the z coordinate to 30 Å and the z velocity to -0.1 Å/fs::

   deposit 10000 2 30 file add_atoms.txt -0.1

Example 5
---------

Deposit atoms read from the file :attr:`add_atoms.txt` every 10000 steps, keeping the positions and velocities exactly as given in the file (:attr:`direction` is :attr:`-1` and no :attr:`velocity` follows the file name)::

   deposit 10000 -1 0 file add_atoms.txt

Caveats
-------

* This keyword can appear only once in a :ref:`run.in <run_in>` file, and only one :ref:`run keyword <kw_run>` is allowed when it is used.
* The original :attr:`run.in` and :attr:`model.xyz` are saved as :attr:`run.in.original` and :attr:`model.xyz.original`, respectively, and the :attr:`run.in` and :attr:`model.xyz` files will be modified during the simulation.
* If the model has grouping, all the deposited atoms are assigned to new groups (with labels one larger than the largest existing group label for each group method), such that they can be distinguished from the original atoms.
