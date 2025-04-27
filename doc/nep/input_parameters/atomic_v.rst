.. _kw_atomic_v:
.. index::
   single: atomic_v (keyword in nep.in)

:attr:`atomic_v`
================

This keyword sets the mode `\atomic_v` of whether to fit atomic or global quantities for dipole (`model_type = 1`) or polarizability (`model_type = 2`). Only one of atomic and global can be fitted at a time. Fitting both simultaneously is not supported. For the virial tensor (`model_type = 0`), only the global model is supported. 
The syntax is::

  atomic_v <mode>

where :attr:`<mode>` must be an integer that can assume one of the following values.

=====  ===========================
Value  Mode 
-----  ---------------------------
0      fit global tensor (default)
1      fit atomic tensor
=====  ===========================
