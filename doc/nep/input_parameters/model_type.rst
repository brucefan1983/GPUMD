.. _kw_model_type:
.. index::
   single: model_type (keyword in nep.in)

:attr:`model_type`
==================

This keyword allows one to specify the type of model that is being trained.
The syntax is::

  model_type <type_value>

where :attr:`<type_value>` must be an integer that can assume one of the following values.

=====  ===================
Value  Type of model
-----  -------------------
0      potential (default)
1      dipole
2      polarizability
=====  ===================
