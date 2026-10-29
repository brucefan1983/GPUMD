.. _kw_charge_mode:
.. index::
   single: charge_mode (keyword in nep.in)

:attr:`charge_mode`
===================

This legacy keyword remains supported but prints a deprecation warning.
Use ``model qnep1`` or ``model qnep2`` for new inputs; see :ref:`model <kw_model>`.
Do not combine ``charge_mode`` with ``model``, including ``charge_mode 0``.

This keyword specifies the charge mode of a NEP4 potential model to be trained.

The syntax is::

  charge_mode <mode_value> [<flip_charge>]

where :attr:`<mode_value>` must be an integer that can assume one of the following values.

=====  ================================================================================
Value  Model description
-----  --------------------------------------------------------------------------------
0      original NEP without charge
1      qNEP model with charge, including both real- and reciprocal-space contributions
2      qNEP model with charge, including reciprocal-space contribution only
=====  ================================================================================

The optional :ref:`flip_charge <kw_flip_charge>` flag is ``0`` (default) or
``1`` and controls charge-sign reversal when restarting training.

Here, qNEP means NEP with charge (q) [Fan2026]_.
Mode 1 is consistent with the approach first proposed by Song et al. [Song2024b]_.
Mode 2 is consistent with the approach first proposed by Cheng [Cheng2025]_.
To train a qNEP model, we do not need to provide target charges.
One can choose to provide target Born effective charges (:term:`BEC`), but this usually only works for a single matter in a single phase, as we have assumed a constant high-frequency dielectric constant for the whole training dataset.
