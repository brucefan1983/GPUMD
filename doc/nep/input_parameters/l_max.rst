.. _kw_l_max:
.. index::
   single: l_max (keyword in nep.in)

:attr:`l_max`
=============

It sets the angular descriptors, see Sect. II.C of [Fan2022b]_.
The syntax is::

  l_max <l_max_3b> {<has_q_222> <has_q_1111> <has_q_112> <has_q_1122>}

where :attr:`l_max_3b` sets the limit for the three-body descriptors and :attr:`has_q_222`, :attr:`has_q_1111`, :attr:`has_q_112`, and :attr:`has_q_1122` optionally specify which four-body and five-body descriptors are used.

:math:`l_\mathrm{max}^\mathrm{3b}` can take values from 0 to 8, and :attr:`has_q_222`, :attr:`has_q_1111`, :attr:`has_q_112`, and :attr:`has_q_1122` can be zero (not to use) or nonzero (to use).

The default values are :math:`l_\mathrm{max}^\mathrm{3b}=4` and only :attr:`has_q_222` is nonzero.
