.. _kw_spin_order:
.. index::
   single: spin_order (keyword in nep.in)

``spin_order``
===================

This keyword sets the maximum polynomial degree of primitive spin inputs.

Syntax
------

::

  spin_order <order>

Parameters
----------

The range is 1--4 and the default is 2. Order 2 includes spin vectors,
squared magnitudes, and rank-2 spin tensors
when permitted by :ref:`spin_l_max <kw_spin_l_max>`. Order 4 also includes
cubic and quartic spin inputs.

This is not an upper bound on the polynomial degree or body order of the
energy: squared tensor norms and the nonlinear fitting network combine
these inputs. See :ref:`nep_spin_contractions`.

Output
------

None.

Examples
--------

::

  spin_order 2

Notes
-----

None.
