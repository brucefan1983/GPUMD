.. _kw_initial_para:
.. index::
   single: initial_para (keyword in nep.in)

:attr:`initial_para`
====================

Set the reference parameter scale used when computing descriptor normalization
factors (``q_scaler``) from the training structures::

  initial_para <value>

The value must be finite and within [0.1, 1]. The default is 1.
This keyword does not set the SNES search mean, random seed, or search width
(``sigma0``).

For ordinary structural NEP, the reference parameter vector is filled with
this value. For Spin3, this value multiplies the structural and magnetic
radial coefficients of a fixed, non-degenerate reference model. Its channel
mixing matrices are retained unchanged so that distinct magnetic channels
remain distinct. ``initial_para 1`` preserves the default Spin3 reference.
The magnetic scaler still obeys the gain cap described in :ref:`nep_spin`;
changing this value does not uniformly rescale every descriptor component.

For example, a fresh Spin3 training run can use::

  spin_version 1
  initial_para 0.5

Spin3 prediction, restart, and :ref:`import_q_scaler 1 <kw_import_q_scaler>`
reuse the scaler saved in ``nep.txt``, regardless of ``initial_para``.
Changing this keyword in those runs does not rescale stored model parameters
or recompute normalization. To compute a new scaler, start a fresh training
run without ``nep.restart`` and with ``import_q_scaler 0``.
