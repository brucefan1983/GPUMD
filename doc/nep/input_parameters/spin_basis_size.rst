.. _kw_spin_basis_size:
.. index::
   single: spin_basis_size (keyword in nep.in)

:attr:`spin_basis_size`
=======================

This keyword sets the largest radial and angular basis indices of the spin
descriptor::

  spin_basis_size <radial_size> <angular_size>

The default is ``8 0``. As for the ordinary NEP basis, a maximum index
:math:`K` represents :math:`K+1` functions because indexing starts from zero.

For ``spin_version 1``, only ``8 0`` is supported: nine radial basis functions and no separate angular basis.
