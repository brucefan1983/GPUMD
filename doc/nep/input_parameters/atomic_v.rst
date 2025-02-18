.. _kw_atomic_v:
.. index::
   single: atomic_v (keyword in nep.in)

:attr:`atomic_v`
================

This keyword sets the mode :math:`\atomic_v` of whether to fit atomic or global quantities for virial (`model_type = 0`), dipole (`model_type = 1`), or polarizability (`model_type = 2`).
The syntax is::

  atomic_v <mode>

where :attr:`<mode>` must be an integer that can assume one of the following values.

=====  ===========================
Value  Mode 
-----  ---------------------------
0      fit global tensor (default)
1      fit atomic tensor
=====  ===========================
