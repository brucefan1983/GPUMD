.. _kw_dump_force:
.. index::
   single: dump_force (keyword in run.in)

:attr:`dump_force`
==================

Write the atomic forces to :ref:`force.out output file <force_out>`.

Syntax
------

.. code::

   dump_force interval <options>

The :attr:`interval` parameter is the output interval (number of steps) of the atom forces.
At the moment, :attr:`<options>` can only assume the value :attr:`group`.
The option :attr:`group` shoud have two parameters::

  group grouping_method group_id

which means only dumping forces of atoms in group :attr:`group_id` within the grouping method :attr:`grouping_method`.
If this option is not used, the forces will be written for all the atoms.

Examples
--------

Example 1
^^^^^^^^^

To dump all the forces every 10 steps for a run, one can add::

  dump_force 10

before the :ref:`run keyword <kw_run>`.

Example 2
^^^^^^^^^

Similar to the above example, but only for atoms in group 1 within grouping method 2::

  dump_force 10 group 2 1
