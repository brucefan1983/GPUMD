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
* In the second usage, a series of forces specified in the file :attr:`add_force_file` will be periodically added to each selected atom. The file format is::

    N
    Fx(0) Fy(0) Fz(0)
    Fx(1) Fy(1) Fz(1)
    ...
    Fx(N-1) Fy(N-1) Fz(N-1)

  The first line contains the number of force vectors :math:`N`, which must be at least 2. It is followed by :math:`N` lines, each containing the three Cartesian components of one force vector.
  At the initial force evaluation (:math:`t=0`), vector 0 is used. At :math:`t=n\Delta t`, vector :math:`n \bmod N` is used, so the sequence is repeated periodically.
* Force is in units of eV/Å.

Example 1
---------

Add a constant force of 0.1 eV/Å in the x direction on atoms in group 1 of group method 2::

   add_force 2 1 0.1 0 0

Example 2
---------

Add force on atoms in group 2 of group method 0 using a sequence of force vectors from :attr:`add_force.txt`::

   add_force 0 2 add_force.txt

For example, the following file applies three force vectors periodically::

   3
   0.10 0.00 0.00
   0.00 0.10 0.00
   0.00 0.00 0.10

Note
----

This keyword can be used multiple times during a run.
