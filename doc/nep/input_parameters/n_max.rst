.. _kw_n_max:
.. index::
   single: n_max (keyword in nep.in)

:attr:`n_max`
=============

This keyword sets the number of radial (:math:`n_\mathrm{max}^\mathrm{R}`) and angular (:math:`n_\mathrm{max}^\mathrm{A}`) descriptor components as introduced in Sect. II.B and Eq. (2) of [Fan2022b]_.
The syntax is::

  n_max <n_max_R> <n_max_A>

where :attr:`n_max_R` and :attr:`n_max_A` are :math:`n_\mathrm{max}^\mathrm{R}` and :math:`n_\mathrm{max}^\mathrm{A}`, respectively.
The two parameters must satisfy :math:`0 \leq n_\mathrm{max}^\mathrm{R},n_\mathrm{max}^\mathrm{A} \leq 19`.

The default values of :math:`n_\mathrm{max}^\mathrm{R}=4` and :math:`n_\mathrm{max}^\mathrm{A}=4` are relatively small but typically yield high speed.

**Note:** These parameters should not be confused with :math:`N_\mathrm{bas}^\mathrm{R}` and :math:`N_\mathrm{bas}^\mathrm{A}`, which are set via the :ref:`basis_size keyword <kw_basis_size>`.
