.. _kw_spin_order:
.. index::
   single: spin_order (keyword in nep.in)

:attr:`spin_order`
===================

This Spin3 keyword selects the highest contraction level supplied to the
neural network::

  spin_order <order>

The value can be 1, 2, or 3 and defaults to 3. It is used only with
``spin_version 1``.

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Value
     - Descriptor content
   * - 1
     - local spin invariants and direct one-edge contractions
   * - 2
     - order 1 plus edge powers, density norms, and two-leg correlations
   * - 3
     - order 2 plus nested three-leg contractions

The levels are nested: order 1 is a prefix of order 2, which is a prefix of
order 3. ``spin_order`` is **not** the body order of the final energy. The
neural network is nonlinear and can combine these coordinates with each other
and with the structural NEP descriptor. See :ref:`nep_spin_contractions` for
the explicit construction.
