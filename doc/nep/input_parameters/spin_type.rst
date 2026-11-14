.. _kw_spin_type:
.. index::
   single: spin_type (keyword in nep.in)

:attr:`spin_type`
======================

Select the atom types that have independent spin degrees of freedom.

Syntax
------

::

  spin_type <species1> [<species2> ...]

Parameters
----------

At least one type must be listed. Names must be unique and must appear in
:ref:`type <kw_type>`. If omitted, all atom types are selected. The effective
selection is saved in the model and restored for prediction and restart;
an explicit conflicting selection is rejected.

Selected types have active spin coordinates and contribute to magnetic losses when labels are present.
All atom types contribute to the geometric environment and retain their atomic forces.
For excluded types, effective spins and magnetic forces are zero, and magnetic-force labels are ignored.
The original data files need no modification.

The grouped-response path derivative uses the selected DFT spin coordinates.
With :ref:`lambda_spin_response <kw_lambda_spin_response>`, derived tangents
are zero on excluded types.

Output
------

None.

Examples
--------

::

  spin_type Fe

Notes
-----

None.
