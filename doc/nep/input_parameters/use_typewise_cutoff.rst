.. _kw_use_typewise_cutoff:
.. index::
   single: use_typewise_cutoff (keyword in nep.in)

:attr:`use_typewise_cutoff`
===========================

This keyword enables one to use typewise cutoff radii for the radial and angular descriptors of the :term:`NEP` model.
The syntax is::

  use_typewise_cutoff [<radial_factor> <angular_factor>]

without two optional (dimensionless) parameters, :attr:`<radial_factor>` and :attr:`<angular_factor>`, which default to 2.5 and 2, respectively.

If this keyword is present, the radial cutoff between two elements is the minimum between the global radial cutoff :math:`r_\mathrm{c}^\mathrm{R}` and :attr:`<radial_factor>` times of the sum of the covalent radii of the two elements, and the angular cutoff between two elements is the minimum between the global angular cutoff :math:`r_\mathrm{c}^\mathrm{A}` and :attr:`<angular_factor>` times of the sum of the covalent radii of the two elements.

By default, this keyword is not in effect.

