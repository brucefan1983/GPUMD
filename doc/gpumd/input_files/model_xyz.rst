.. _model_xyz:
.. index::
   single: gpumd input files; Simulation model
   single: gpumd input files; Atomic configuration
   single: model.xyz

Simulation model (``model.xyz``)
================================

The ``model.xyz`` file supplies the initial atomic configuration, simulation cell, and boundary conditions.
It uses the `extended XYZ format <https://github.com/libAtoms/extxyz>`_: one atom-count line, one header line, and one line per atom.

File format
-----------

Line 1: atom count
^^^^^^^^^^^^^^^^^^

Give the number of atoms, :math:`N`, as a single integer.

Line 2: header
^^^^^^^^^^^^^^

The header contains space-separated ``keyword=value`` pairs.
Enclose a value containing several items in double quotes.
Header text is case-insensitive, and spaces around ``=`` or inside the quotes are allowed.
GPUMD reads the following fields; other header fields are ignored.

* ``lattice="ax ay az bx by bz cx cy cz"`` (required): the Cartesian components of the three cell vectors, in Å:

  .. math::

     \boldsymbol{a} = a_x \boldsymbol{e}_x + a_y \boldsymbol{e}_y + a_z \boldsymbol{e}_z\\
     \boldsymbol{b} = b_x \boldsymbol{e}_x + b_y \boldsymbol{e}_y + b_z \boldsymbol{e}_z\\
     \boldsymbol{c} = c_x \boldsymbol{e}_x + c_y \boldsymbol{e}_y + c_z \boldsymbol{e}_z

* ``properties=property_name:data_type:number_of_columns:...`` (required): the names, types, and column counts of the per-atom data.
  Join successive properties with colons, in the order used on each atom line.
  ``S``, ``R``, and ``I`` denote strings, real numbers, and integers.
* ``pbc="pbc_a pbc_b pbc_c"`` (optional): ``T`` for periodic or ``F`` for free boundaries along each cell vector.
  The default is ``pbc="T T T"``.

Line 3 onward: per-atom data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Provide exactly :math:`N` atom lines, each with the columns declared by ``properties``.
The supported properties are:

.. list-table::
   :header-rows: 1
   :widths: 27 26 12 35

   * - Property
     - Quantity
     - Unit
     - Requirement or default
   * - ``species:S:1``
     - Chemical symbol
     -
     - Required
   * - ``pos:R:3``
     - Cartesian position
     - Å
     - Required
   * - ``mass:R:1``
     - Atomic mass
     - amu
     - Default mass for the element
   * - ``charge:R:1``
     - Atomic charge
     - e
     - Zero
   * - ``vel:R:3``
     - Cartesian velocity
     - Å/fs
     - Optional
   * - ``spin:R:3``
     - Magnetic moment
     - muB
     - Required for NEP-spin
   * - ``spin_vel:R:3``
     - Spin velocity
     - muB/fs
     - Optional; requires ``spin:R:3``
   * - ``group:I:M``
     - Group labels
     -
     - Optional; :math:`M` grouping methods

Here, amu is the unified atomic mass unit, e is the elementary charge, and muB is the Bohr magneton.
When ``vel`` is present, GPUMD uses these velocities and ignores the initial temperature specified by :ref:`velocity <kw_velocity>`.
Otherwise, that command sets the initial temperature, which defaults to 300 K.

.. _grouping_methods:

Grouping methods
----------------

``group:I:M`` supplies one group label per atom for each of :math:`M` independent grouping methods, numbered from 0 to :math:`M-1` in column order.
Within each method, every atom belongs to the group identified by its label, which must be an integer from 0 to :math:`N-1`.
The method number selects a partition of the atoms; the group label selects one group within that partition.

These labels remain attached to the atoms as they move; group membership is not recalculated from their current positions.
If the ``group`` property is omitted, no grouping methods are defined.

Example
-------

This example illustrates the file format and three grouping methods::

   10
   pbc="T F F" lattice="10 0 0 0 1 0 0 0 1" properties=species:S:1:pos:R:3:group:I:3
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


The ten atoms alternate between C and Si along x, at positions from 0 to 9 Å.
The cell is 10 Å by 1 Å by 1 Å, with periodic boundaries only along x.
Default masses are used, and velocities are not supplied.
The last three columns define:

* Method 0: atoms 0-4 belong to group 0, and atoms 5-9 belong to group 1.
* Method 1: atom :math:`m` belongs to group :math:`m`, so each group has one atom.
* Method 2: all atoms belong to group 0.

Notes
-----

For non-NEP potentials, the cell thickness in each periodic direction must exceed twice the potential cutoff.
NEP potentials include the periodic images needed for smaller cells.

Magnetic force is the negative energy derivative with respect to spin, in eV/muB.
The spin torque, spin crossed with magnetic force, is in eV.
External magnetic induction B is specified in tesla (T).
Spin vectors store Cartesian magnetic moments, not normalized directions.
Active spins are not normalized; training and runtime data must use the same
spin convention. Elements excluded by the model's ``spin_type`` have zero
effective spin and zero magnetic force, while their geometry is retained.
For NEP-spin models, the ``spin`` property may not be repeated, and
``spin:R:3`` is required; missing spin values are never filled with zeros.
Other potential models ignore the ``spin`` property.
