.. _kw_model_type:
.. index::
   single: model_type (keyword in nep.in)

:attr:`model_type`
==================

This legacy keyword remains supported but prints a deprecation warning.
Use :ref:`model <kw_model>` for new inputs. Do not combine the two keywords.

This keyword specifies the type of model that is being trained.
The syntax is::

  model_type <type_value>

where :attr:`<type_value>` must be an integer that can assume one of the following values.

=====  ============================
Value  Type of model
-----  ----------------------------
0      potential (default)
1      dipole
2      polarizability
3      temperature-dependent energy
=====  ============================
