.. _kw_fix:
.. index::
   single: fix (keyword in run.in)

:attr:`fix`
===========

Freeze a group of atoms by setting their velocities and forces to zero.

Syntax
------
::

  fix <group_label>
  fix <grouping_method> <group_label>

Parameters
----------

- If only :attr:`group_label` is given, grouping method 0 is used by default.
- If :attr:`grouping_method` is also specified, the given grouping method defined in the :ref:`simulation model file <model_xyz>` will be used.

Examples
--------

Fix group 0 using the default grouping method 0::

  fix 0

Fix group 2 using grouping method 1::

  fix 1 2

Notes
-----
* This command applies to the current run. Repeat it before each run in which the group should remain fixed.
* When used with :ref:`move <kw_move>`, the fixed and moving groups must be different and must use the same grouping method.
