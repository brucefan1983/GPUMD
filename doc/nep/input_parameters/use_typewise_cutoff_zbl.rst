.. _kw_cutoff:
.. index::
   single: use_typewise_cutoff_zbl (keyword in nep.in)

:attr:`use_typewise_cutoff_zbl`
===============================

This keyword enables one to use typewise cutoff radii for the ZBL part of the :term:`NEP` model.
The syntax is::

  use_typewise_cutoff_zbl

without any parameter.

If this keyword is present, the outer ZBL cutoff between elements I and J is the minimum between the global ZBL outer cutoff :math:`r_\mathrm{outer}^\mathrm{ZBL}` and 0.6 times of the sum of the covalent radii of the two elements, and the inner ZBL cutoff is half of the outer one.

By default, this keyword is not in effect.
