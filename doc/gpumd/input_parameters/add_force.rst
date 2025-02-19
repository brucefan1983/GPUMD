.. _kw_add_force:
.. index::
   single: add_force (keyword in run.in)

:attr:`add_force`
=================

This keyword is used to add force on atoms in a selected group at each step during a run.

Syntax
------

This keyword is used in one of the following two ways::

  add_force <group_method> <group_id> <Fx> <Fy> <Fz> # usage 1
  add_force <group_method> <group_id> <add_force_file> # usage 2

* Force is added to atoms in group :attr:`group_id` of group method :attr:`group_method`.
* In the first usage, the constant force with components :attr:`Fx`, :attr:`Fy`, and :attr:`Fz` is added on each selected atom.
* In the second usage, a series of forces specified in the file :attr:`add_force_file` will be periodically added on each selected atom.
* Force is in units of eV/Å.

Example 1
---------

Add a constant force of 0.1 eV/Å in the x direction on atoms in group 1 of group method 2::

   add_force 2 1 0.1 0 0

Example 2
---------

Add force on atoms in group 2 of group method 0, where the file :attr:`add_force.txt` contains a number of rows of 3 values specifying a sequence of force vectors to be periodically applied during the run::

   add_force 0 2 add_force.txt

Note
----

This keyword can be used multiple times during a run.
