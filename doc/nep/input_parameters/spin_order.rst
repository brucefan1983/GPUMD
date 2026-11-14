.. _kw_spin_order:
.. index::
   single: spin_order (keyword in nep.in)

:attr:`spin_order`
====================

Set the maximum polynomial degree of primitive spin inputs.

Syntax
------

::

  spin_order S

Parameters
----------

The range is 1--4 and the default is 2. At degree :math:`d`, primitive ranks
are :math:`j=d\bmod2,d\bmod2+2,\ldots,\min(d,L)`, where :math:`L` is
:ref:`spin_l_max <kw_spin_l_max>`. Order 2 includes the spin vector,
squared magnitude and rank-2 spin tensor when the rank limit permits them.
Order 4 also includes cubic and quartic primitives.

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

See :ref:`nep_spin` for the descriptor definition and model contract.
