.. _kw_type:
.. index::
   single: mode (keyword in nep.in)

:attr:`mode`
============

This keyword allows one to specify the type of model that is being trained.
The syntax is::

  mode <mode_value>

where :attr:`<mode_value>` must be an integer that can assume one of the following values.

=====  ===================
Value  Type of model
-----  -------------------
0      potential (default)
1      dipole
2      polarizability
=====  ===================
