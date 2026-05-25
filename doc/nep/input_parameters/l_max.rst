.. _kw_l_max:
.. index::
   single: l_max (keyword in nep.in)

:attr:`l_max`
=============

It sets the angular descriptors, see Sect. II.C of [Fan2022b]_.
The syntax is::

  l_max <l_max_3b> {<has_q_222> <has_q_1111> <has_q_112> <has_q_123> <has_q_233>}

where :attr:`l_max_3b` sets the limit for the three-body descriptors, :attr:`has_q_222`, :attr:`has_q_112`, :attr:`has_q_123`, and :attr:`has_q_233` optionally specify which four-body descriptors are used, and :attr:`has_q_1111` specifies if the five-body descriptor is used.

:math:`l_\mathrm{max}^\mathrm{3b}` can take values from 2 to 8, and :attr:`has_q_222`, :attr:`has_q_1111`, :attr:`has_q_112`, :attr:`has_q_123`, and :attr:`has_q_233` can be 0 (not to use) or 1 (to use).

The default values are::

  l_max 4 1 0 0 0 0 # equivalent to l_max 4 1 or the old-style l_max 4 2
  
For high accuracy, one can try::
  l_max 4 1 0 1 1 1

The five-body descriptor switch :attr:`has_q_1111` is only provided for backward compatibility. 
This five-body descriptor is equivalent to the square of two three-body descriptors and thus does not provide new information. It use is strongly discouraged.
