.. _kw_fix:
.. index::
   single: fix (keyword in run.in)

:attr:`fix`
===========

This keyword can be used to fix (freeze) a group of atoms

Syntax
------
This keyword requires a single parameter which is the label of the group in which the atoms are to be fixed (velocities and forces are always set to zero such that the atoms in the group do not move).
The full command reads::

  fix <group_label>

Here, the :attr:`group_label` refers to the grouping method 0 defined in the :ref:`simulation model file <model_xyz>`.

Example
-------
The use of this keyword is illustrated in the :ref:`tutorial on thermal transport from NEMD and HNEMD simulations <tutorials>`.

Caveats
-------
This keyword is not propagating, which means that it only affects the simulation within the run it belongs to.
