.. _kw_cutoff:
.. index::
   single: cutoff (keyword in nep.in)

:attr:`cutoff`
==============

This keyword enables one to specify the radial (:math:`r_\mathrm{c}^\mathrm{R}`) and angular (:math:`r_\mathrm{c}^\mathrm{A}`) cutoffs of the :term:`NEP` model.
The syntax is::

  cutoff <radial_cutoff> <angular_cutoff>

where :attr:`<radial_cutoff>` and :attr:`<angular_cutoff>` correspond to :math:`r_\mathrm{c}^\mathrm{R}` and :math:`r_\mathrm{c}^\mathrm{A}`, respectively.
The cutoffs must satisfy the conditions 2.5 Å :math:`\leq r_\mathrm{c}^\mathrm{A} \leq r_\mathrm{c}^\mathrm{R} \leq` 10 Å.

The defaults are :math:`r_\mathrm{c}^\mathrm{R}` = 8 Å and :math:`r_\mathrm{c}^\mathrm{A}` = 4 Å.
It can be computationally beneficial to use (possibly much) smaller :math:`r_\mathrm{c}^\mathrm{R}` but the default values should be reasonable in most cases.
