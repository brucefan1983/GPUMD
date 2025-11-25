.. _kw_charge_mode:
.. index::
   single: charge_mode (keyword in nep.in)

:attr:`charge_mode`
===================

This keyword allows one to specify the charge mode of a NEP4 potential model to be trained.

The syntax is::

  charge_mode <mode_value>

where :attr:`<mode_value>` must be an integer that can assume one of the following values.

=====  ================================================================================
Value  Model description
-----  --------------------------------------------------------------------------------
0      original NEP without charge
1      qNEP model with charge, including both real- and reciprocal-space contributions
2      qNEP model with charge, including reciprocal-space contribution
=====  ================================================================================

Here, qNEP means NEP with charge (q). 
To train a qNEP model, we do not need to provide target charges.
One can choose to provide target Born effective charges (:term:`BEC`), but this usually only works for a single matter in a single phase, as we have assumed a constant high-frequency dielectric constant for the whole training dataset.
