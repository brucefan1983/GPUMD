.. _kw_spin_cutoff:
.. index::
   single: spin_cutoff (keyword in nep.in)

:attr:`spin_cutoff`
====================

This keyword sets the Spin3 cutoff distance in Å::

  spin_cutoff <cutoff>
  spin_cutoff <cutoff_type_1> ... <cutoff_type_N>

Use either one positive value, which is broadcast to all atom types, or exactly
one positive value for every type in the order declared by :ref:`type
<kw_type>`. If this keyword is absent, the largest ordinary radial
:ref:`cutoff <kw_cutoff>` is broadcast.

Every value must be less than or equal to the largest ordinary radial cutoff
(``rc_radial_max``). The same restriction applies when reading a model
file. To use a larger magnetic cutoff, increase the ordinary radial cutoff
as well.

For an edge between types :math:`a` and :math:`b`, Spin3 uses the arithmetic mean
:math:`(r_a^\mathrm{spin}+r_b^\mathrm{spin})/2`.

For example, with ``type 2 Fe Ge``::

  cutoff 6.0 5.0
  spin_cutoff 4.0 6.0

uses 4, 5, and 6 Å for Fe--Fe, Fe--Ge, and Ge--Ge magnetic edges. The ``spin_cutoff``
line does not change the ordinary structural NEP cutoffs.
