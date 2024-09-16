.. _kw_use_typewise_cutoff_zbl:
.. index::
   single: use_typewise_cutoff_zbl (keyword in nep.in)

:attr:`use_typewise_cutoff_zbl`
===============================

This keyword enables one to use typewise cutoff radii for the ZBL part of the :term:`NEP` model.
The syntax is::

  use_typewise_cutoff_zbl [<factor>]

with one optional (dimensionless) parameter :attr:`<factor>` that defaults to 0.65.

If this keyword is present, the outer ZBL cutoff between two elements is the minimum between the global outer ZBL cutoff :math:`r_\mathrm{outer}^\mathrm{ZBL}` and :attr:`<factor>` times of the sum of the covalent radii of the two elements, and the inner ZBL cutoff is half of the outer one.

By default, this keyword is not in effect.
