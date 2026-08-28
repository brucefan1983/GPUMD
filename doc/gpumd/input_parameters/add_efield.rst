.. _kw_add_efield:
.. index::
   single: add_efield (keyword in run.in)

:attr:`add_efield`
==================

This keyword is used to add force due to electric field on atoms in a selected group at each step during a run. 

For qNEP models, the force equals to the dot product of the electric field and the Born effective charge (:term:`BEC`). 
This is only meaningful when the qNEP model was trained with target :term:`BEC`.

For other models, the force equals to the product of the electric field and the charge of the atom as specified in :attr:`model.xyz` via :attr:`charge:R:1`.

Syntax
------

This keyword is used in one of the following ways::

  add_efield <group_method> <group_id> <Ex> <Ey> <Ez> # usage 1
  add_efield <group_method> <group_id> <add_efield_file> # usage 2
  add_efield <group_method> <group_id> <Ex> <Ey> <Ez> <mode> # usage 3
  add_efield <group_method> <group_id> <add_efield_file> <mode> # usage 4

* Electric field is applied to atoms in group :attr:`group_id` of group method :attr:`group_method`.
* In usage 1, the constant electric field with components :attr:`Ex`, :attr:`Ey`, and :attr:`Ez` is applied to each selected atom.
* In usages 2 and 4, a series of electric fields specified in the file :attr:`add_efield_file` will be periodically applied to each selected atom. The file format is::

    N
    Ex(0) Ey(0) Ez(0)
    Ex(1) Ey(1) Ez(1)
    ...
    Ex(N-1) Ey(N-1) Ez(N-1)

  The first line contains the number of electric-field vectors :math:`N`, which must be at least 2. It is followed by :math:`N` lines, each containing the three Cartesian components of one electric-field vector.
  At the initial force evaluation (:math:`t=0`), vector 0 is used. At :math:`t=n\Delta t`, vector :math:`n \bmod N` is used, so the sequence is repeated periodically.
* In usages 1 and 2, if the potential model is qNEP, the added electric force equals to the dot product of the electric field and the :term:`BEC`; otherwise it equals to the product of the electric field and the charge of the atom as specified in :attr:`model.xyz` via :attr:`charge:R:1`.
* In usages 3 and 4, :attr:`mode` can be charge or bec.
  * When :attr:`mode` is charge, the electric force will be calculated via 

    * the qNEP predicted charges for qNEP potential models
    * the user-specified charges for other potential models
    
  * When :attr:`mode` is bec, the potential model must be qNEP, and the :term:`BEC` will be used to calculate the electric force.
* Electric field is in units of V/Å.

Example 1
---------

Add a constant electric field of 0.1 V/Å in the x direction on atoms in group 1 of group method 2::

   add_efield 2 1 0.1 0 0

Example 2
---------

Add electric field on atoms in group 2 of group method 0 using a sequence of electric-field vectors from :attr:`add_efield.txt`::

   add_efield 0 2 add_efield.txt

For example, the following file applies three electric-field vectors periodically::

   3
   0.10 0.00 0.00
   0.00 0.10 0.00
   0.00 0.00 0.10

Note
----

This keyword can be used multiple times during a run.
