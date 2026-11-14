.. _kw_basis_size:
.. index::
   single: basis_size (keyword in nep.in)

:attr:`basis_size`
==================

Set the maximum basis indices used to construct the radial and angular descriptor functions; see Sects. II.B and II.C and Eq. (3) of [Fan2022b]_.

Syntax
------

::

  basis_size <N_bas_R> <N_bas_A>

Parameters
----------

``N_bas_R`` and ``N_bas_A`` set :math:`N_\mathrm{bas}^\mathrm{R}` and :math:`N_\mathrm{bas}^\mathrm{A}`, respectively.
Both default to 6, which is usually sufficient.
The recommended range for each parameter is 0 to 8.
The supported bounds are :math:`0 \leq N_\mathrm{bas}^\mathrm{R} \leq 16` and :math:`0 \leq N_\mathrm{bas}^\mathrm{A} \leq 12`.
A maximum index :math:`K` represents :math:`K+1` basis functions.

For ``model nep_spin``, the second value also sets the maximum radial basis
index for the spin descriptor. For example, ``basis_size 8 8`` gives nine
functions (indices 0 through 8). The spin descriptor reuses the structural
angular radial coefficients with its own :ref:`spin_cutoff <kw_spin_cutoff>`.

Output
------

None.

Examples
--------

::

  basis_size 8 8

Notes
-----

These basis sizes are distinct from the descriptor index limits set by :ref:`n_max <kw_n_max>`.
