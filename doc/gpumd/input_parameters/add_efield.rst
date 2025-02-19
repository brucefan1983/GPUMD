.. _kw_add_efield:
.. index::
   single: add_efield (keyword in run.in)

:attr:`add_efield`
==================

This keyword is used to add force due to electric field on atoms in a selected group at each step during a run. The force equals to the product of the electric field and the charge of the atom as specified in :attr:`model.xyz` via :attr:`charge:R:1`.

Syntax
------

This keyword is used in one of the following two ways::

  add_efield <group_method> <group_id> <Ex> <Ey> <Ez> # usage 1
  add_efield <group_method> <group_id> <add_efield_file> # usage 2

* Electric field is applied to atoms in group :attr:`group_id` of group method :attr:`group_method`.
* In the first usage, the constant electric field with components :attr:`Ex`, :attr:`Ey`, and :attr:`Ez` is applied to each selected atom.
* In the second usage, a series of electric fields specified in the file :attr:`add_efield_file` will be periodically applied to each selected atom.
* Electric field is in units of V/Å.

Example 1
---------

Add a constant electric field of 0.1 V/Å in the x direction on atoms in group 1 of group method 2::

   add_efield 2 1 0.1 0 0

Example 2
---------

Add electric field on atoms in group 2 of group method 0, where the file :attr:`add_efield.txt` contains a number of rows of 3 values specifying a sequence of electric field vectors to be periodically applied during the run::

   add_efield 0 2 add_efield.txt

Note
----

This keyword can be used multiple times during a run.
