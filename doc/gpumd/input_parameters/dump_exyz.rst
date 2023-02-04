.. _kw_dump_exyz:
.. index::
   single: dump_exyz (keyword in run.in)

:attr:`dump_exyz`
=================

Write some data into dump.xyz in `extended XYZ format <https://github.com/libAtoms/extxyz>`_.

Syntax
------

.. code::

   dump_exyz <interval> <has_velocity> <has_force>

Here, the :attr:`interval` parameter is the output interval (number of steps) of the data.
:attr:`has_velocity` can be 1 or 0, which means the velocities will or will not be included in the output.
:attr:`has_force` can be 1 or 0, which means the forces will or will not be included in the output.

Examples
--------

To dump the positions, velocities, and forces every 1000 steps for a run, one can add::

  dump_exyz 1000 1 1

before the :ref:`run keyword <kw_run>`.

Caveats
-------
* This keyword is not propagating.
  That means, its effect will not be passed from one run to the next.
* The output file has an appending behavior and will result in a single `dump.xyz` file no matter how many times the simulation is run.
