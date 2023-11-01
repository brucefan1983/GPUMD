.. _kw_dump_exyz:
.. index::
   single: dump_exyz (keyword in run.in)

:attr:`dump_exyz`
=================

Write some data into dump.xyz in `extended XYZ format <https://github.com/libAtoms/extxyz>`_.

Syntax
------

.. code::

   dump_exyz <interval> <has_velocity>
   dump_exyz <interval> <has_velocity> <has_force>
   dump_exyz <interval> <has_velocity> <has_force> <has_potential>

Here, the :attr:`interval` parameter is the output interval (number of steps) of the data.
:attr:`has_velocity` can be 1 or 0, which means the velocities will or will not be included in the output.
:attr:`has_force` can be 1 or 0, which means the forces will or will not be included in the output.
:attr:`has_potential` can be 1 or 0, which means the atomic potential energies will or will not be included in the output.
The atomic positions will always be included in the output.

Examples
--------
    # ensemble keyword here
    dump_exyz 1000        # dump positions every 1000 steps
    dump_exyz 1000 1      # dump positions and velocities
    dump_exyz 1000 1 1    # dump positions, velocities, and forces
    dump_exyz 1000 1 1 1  # dump positions, velocities, forces, and potentials
    dump_exyz 1000 0 1 1  # dump positions, forces and potentials
    run 1000000

Caveats
-------
* This keyword is not propagating.
  That means, its effect will not be passed from one run to the next.
* The output file has an appending behavior and will result in a single `dump.xyz` file no matter how many times the simulation is run.
