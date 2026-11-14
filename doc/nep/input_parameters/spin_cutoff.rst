.. _kw_spin_cutoff:
.. index::
   single: spin_cutoff (keyword in nep.in)

:attr:`spin_cutoff`
====================

This keyword sets the spin cutoff distance in Å::

  spin_cutoff <cutoff>
  spin_cutoff <cutoff_type_1> ... <cutoff_type_N>

Set the :ref:`type keyword <kw_type>` before this keyword.

Use either one value of at least 3 Å, which is broadcast to all atom types, or exactly
one value of at least 3 Å for every type in the order declared by :ref:`type
<kw_type>`. If this keyword is absent, the largest ordinary radial
:ref:`cutoff <kw_cutoff>` is broadcast.

Each value must be finite and at least 3 Å, matching the lower bound of the
ordinary angular cutoff. This restriction applies to both training inputs and
model files, including each entry in the per-type form.
Models with a magnetic cutoff below 3 Å are rejected.

Every value must be less than or equal to the largest ordinary radial cutoff
(``rc_radial_max``). The same restriction applies when reading a model
file. To use a larger magnetic cutoff, increase the ordinary radial cutoff
as well.

For an edge between types :math:`a` and :math:`b`, spin uses the arithmetic mean
:math:`(r_a^\mathrm{spin}+r_b^\mathrm{spin})/2`.

For example, with ``type 2 Fe Ge``::

  cutoff 6.0 5.0
  spin_cutoff 4.0 6.0

uses 4, 5, and 6 Å for Fe--Fe, Fe--Ge, and Ge--Ge magnetic edges. The ``spin_cutoff``
line does not change the ordinary structural NEP cutoffs.
