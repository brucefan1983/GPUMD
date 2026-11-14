.. _kw_basis_size:
.. index::
   single: basis_size (keyword in nep.in)

:attr:`basis_size`
==================

It sets the maximum basis indices used to build the radial and angular descriptor functions, see Sects. II.B and II.C as well as Eq. (3) in [Fan2022b]_.
The syntax is::

  basis_size <N_bas_R> <N_bas_A>

where :attr:`<N_bas_R>` and :attr:`<N_bas_A>` set :math:`N_\mathrm{bas}^\mathrm{R}` and :math:`N_\mathrm{bas}^\mathrm{A}`, respectively.
The recommended ranges are :math:`0 \leq N_\mathrm{bas}^\mathrm{R} \leq 8` and :math:`0 \leq N_\mathrm{bas}^\mathrm{A} \leq 8`.

The parameters must satisfy :math:`0 \leq N_\mathrm{bas}^\mathrm{R} \leq 16` and :math:`0 \leq N_\mathrm{bas}^\mathrm{A} \leq 12`.

The default values of :math:`N_\mathrm{bas}^\mathrm{R}=6` and :math:`N_\mathrm{bas}^\mathrm{A}=6` are usually sufficient.

For ``model nep_spin``, the second value also sets the maximum radial basis
index for the spin descriptor. For example, ``basis_size 8 8`` gives nine
functions (indices 0 through 8). The spin descriptor reuses the structural
angular radial coefficients with its own :ref:`spin_cutoff <kw_spin_cutoff>`.

**Note:** These parameters should not be confused with :math:`n_\mathrm{max}^\mathrm{R}` and :math:`n_\mathrm{max}^\mathrm{A}`, which are set via the :ref:`n_max keyword <kw_n_max>`.
