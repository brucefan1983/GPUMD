.. _kw_basis_size:
.. index::
   single: basis_size (keyword in nep.in)

:attr:`basis_size`
==================

It sets the number of basis functions that are used to build the radial and angular descriptor functions, see Sects. II.B and II.C as well as Eq. (3) in [Fan2022b]_.
The syntax is::

  basis_size <N_bas_R> <N_bas_A>

where :attr:`<N_bas_R>` and :attr:`<N_bas_A>` set :math:`N_\mathrm{bas}^\mathrm{R}` and :math:`N_\mathrm{bas}^\mathrm{A}`, respectively.
The parameters must satisfy :math:`0 \leq N_\mathrm{bas}^\mathrm{R},N_\mathrm{bas}^\mathrm{A} \leq 19`.

The default values of :math:`N_\mathrm{bas}^\mathrm{R}=8` and :math:`N_\mathrm{bas}^\mathrm{A}=8` are usually sufficient.

**Note:** These parameters should not be confused with :math:`n_\mathrm{max}^\mathrm{R}` and :math:`n_\mathrm{max}^\mathrm{A}`, which are set via the :ref:`n_max keyword <kw_n_max>`.
