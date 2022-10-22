.. _kw_l_max:
.. index::
   single: l_max (keyword in nep.in)

:attr:`l_max`
=============

This keyword is only relevant for NEP3 and NEP4.
It sets the maximum expansion order for the angular terms, see Sect. II.C of [Fan2022b]_.
The syntax is::

  l_max <l_max_3b> {<l_max_4b> {<l_max_5b>}}

where :attr:`l_max_3b`, :attr:`l_max_4b`, and :attr:`l_max_5b` set the limits for three, four, and five-body terms, respectively.
The latter two arguments are optional (as indicated by the curly brackets).

If there is one value :math:`l_\mathrm{max}^\mathrm{4b}=l_\mathrm{max}^\mathrm{5b}=0`.
If there are two values :math:`l_\mathrm{max}^\mathrm{5b}=0`.

:math:`l_\mathrm{max}^\mathrm{3b}` can only be 4, :math:`l_\mathrm{max}^\mathrm{4b}` can be 0 or 2, and :math:`l_\mathrm{max}^\mathrm{5b}` can be 0 or 1.

The default values are :math:`l_\mathrm{max}^\mathrm{3b}=4`, :math:`l_\mathrm{max}^\mathrm{4b}=2`, and :math:`l_\mathrm{max}^\mathrm{5b}=0`.
