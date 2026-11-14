.. _kw_l_max:
.. index::
   single: l_max (keyword in nep.in)

:attr:`l_max`
=============

It sets the angular descriptors, see Sect. II.C of [Fan2022b]_.

Syntax
------

::

  l_max <l_max_3b> {<has_q_222> <has_q_1111> <has_q_112> <has_q_123> <has_q_233> <has_q_134>}

Parameters
----------

:attr:`l_max_3b` sets the limit for the three-body descriptors, :attr:`has_q_222`, :attr:`has_q_112`, :attr:`has_q_123`, :attr:`has_q_233`, and :attr:`has_q_134` optionally specify which four-body descriptors are used, and :attr:`has_q_1111` specifies if the five-body descriptor is used.

Each enabled four-body or five-body invariant contributes :math:`n_\mathrm{max}^\mathrm{A}+1` descriptor components; see :ref:`Model dimensions <nep_model_dimensions>` for the total descriptor count.

:math:`l_\mathrm{max}^\mathrm{3b}` can take values from 2 to 8, and :attr:`has_q_222`, :attr:`has_q_1111`, :attr:`has_q_112`, :attr:`has_q_123`, :attr:`has_q_233`, and :attr:`has_q_134` can be 0 (not to use) or 1 (to use).

The default is ``l_max 4 1``; the other invariant switches default to 0.

Output
------

None.

Examples
--------

These two inputs select the same default descriptors::

  l_max 4 1
  l_max 4 1 0 0 0 0 0 # equivalent way to write it

Notes
-----

The five-body descriptor switch :attr:`has_q_1111` is only provided for backward compatibility.
This five-body descriptor is equivalent to the square of two three-body descriptors and thus does not provide new information. Its use is strongly discouraged.
