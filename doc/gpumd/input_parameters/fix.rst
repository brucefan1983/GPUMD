.. _kw_fix:
.. index::
   single: fix (keyword in run.in)

:attr:`fix`
===========

This keyword can be used to fix (freeze) a group of atoms

Syntax
------
This keyword accepts 1 or 2 parameters.
The atoms in the specified group will be fixed (velocities and forces are always set to zero such that the atoms do not move).
The full command reads::

  fix <group_label>
  fix <grouping_method> <group_label>

- If only :attr:`group_label` is given, grouping method 0 is used by default.
- If :attr:`grouping_method` is also specified, the given grouping method defined in the :ref:`simulation model file <model_xyz>` will be used.

Example
-------

Fix group 0 using the default grouping method 0::

  fix 0

Fix group 2 using grouping method 1::

  fix 1 2

Caveats
-------
* This keyword is not propagating, which means that it only affects the simulation within the run it belongs to.
* When both :attr:`fix` and :ref:`move <kw_move>` are used, they must use the same grouping method.
