.. _kw_dump_velocity:
.. index::
   single: dump_velocity (keyword in run.in)

:attr:`dump_velocity`
=====================

Dump the atomic velocities to the :ref:`velocity.out output file <velocity_out>`.

Syntax
------

.. code::

   dump_velocity <interval> [{optional_args}]

Here, the :attr:`interval` parameter is the output interval (number of steps) of the atomic velocities.
At the moment, the only optional argument (:attr:`optional_args`) is :attr:`group`.
The option :attr:`group` shoud have two parameters::

  group <grouping_method> <group_id>

which means only dumping velocities of atoms in group :attr:`group_id` within the grouping method :attr:`grouping_method`.
If this option is not used, velocities will be dumped for all the atoms.

Examples
--------

Example 1
^^^^^^^^^
To dump all the velocities every 10 steps for a run, one can add::

  dump_velocity 10

before the :ref:`run keyword <kw_run>`.

Example 2
^^^^^^^^^

Similar to the above example, but only for atoms in group 1 within grouping method 2::

  dump_velocity 10 group 2 1
