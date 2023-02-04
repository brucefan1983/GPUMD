.. _kw_type:
.. index::
   single: train_mode (keyword in nep.in)

:attr:`train_mode`
==================

This keyword allows one to specify the type of model that is being trained.
The syntax is::

  train_mode <mode>

where :attr:`<mode>` must be an integer that can assume one of the following values.

=====  ===================
Value  Type of model
-----  -------------------
0      potential (default)
1      dipole
2      polarizability
=====  ===================
