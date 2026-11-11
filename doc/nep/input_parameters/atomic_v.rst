.. _kw_atomic_v:
.. index::
   single: atomic_v (keyword in nep.in)

:attr:`atomic_v`
================

This keyword selects whether to fit atomic or global quantities for dipoles (``model nep_dipole``) or polarizabilities (``model nep_polarizability``). Only one of atomic and global can be fitted at a time. Fitting both simultaneously is not supported. For the virial tensor (``model nep``), only global fitting is supported.
The syntax is::

  atomic_v <mode>

where :attr:`<mode>` must be an integer that can assume one of the following values.

=====  ===========================
Value  Mode 
-----  ---------------------------
0      fit global tensor (default)
1      fit atomic tensor
=====  ===========================

See :ref:`train.xyz and test.xyz <train_test_xyz>` for the required global and atomic labels.
The output row counts and normalization for the two modes are described under :ref:`dipole_*.out <nep_dipole_out>` and :ref:`polarizability_*.out <nep_polarizability_out>`.
