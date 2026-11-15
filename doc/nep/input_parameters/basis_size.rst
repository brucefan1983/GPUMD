.. _kw_basis_size:
.. index::
   single: basis_size (keyword in nep.in)

:attr:`basis_size`
==================

Set the basis sizes used to construct the radial and angular descriptor functions; see Sects. II.B and II.C and Eq. (3) of [Fan2022b]_.

Syntax
------

::

  basis_size <N_bas_R> <N_bas_A>

Parameters
----------

``N_bas_R`` and ``N_bas_A`` set :math:`N_\mathrm{bas}^\mathrm{R}` and :math:`N_\mathrm{bas}^\mathrm{A}`, respectively.
Both default to 6, which is usually sufficient.
The recommended range for each parameter is 0 to 8.

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
