.. _kw_lambda_m:
.. index::
   single: lambda_m (keyword in nep.in)

:attr:`lambda_m`
=================

This keyword sets the weight of Cartesian magnetic-force RMSE in the loss::

  lambda_m <weight>

The weight must be non-negative and defaults to 1. It multiplies the RMSE of
the Cartesian magnetic force

.. math::

   \boldsymbol{M}_i=-\frac{\partial U}{\partial\boldsymbol{s}_i},

whose unit is eV/muB. Only frames containing ``mforce:R:3`` and
atoms enabled by :ref:`spin_type <kw_spin_type>` enter this RMSE. A
frame without magnetic-force labels can still contribute energy, force, or
virial data.

Set ``lambda_m 0`` to remove the Cartesian magnetic-force RMSE from the
optimization objective. This does not change the analytic magnetic-force
definition or the model format. ``lambda_mforce`` is an accepted input alias,
but ``lambda_m`` is the canonical spelling used in the documentation and
training output.


.. _kw_lambda_m_parallel:
.. _kw_lambda_m_perp:
.. index::
   single: lambda_m_parallel (keyword in nep.in)
   single: lambda_m_perp (keyword in nep.in)

Independent parallel and perpendicular losses
---------------------------------------------

To fit the magnetic-moment magnitude and direction with separate coefficients,
replace ``lambda_m`` with **both** of the following keywords::

  lambda_m_parallel 1.0
  lambda_m_perp     1.0

Each coefficient must be finite and non-negative. Set either coefficient to
zero to disable that component. Mixing these keywords with ``lambda_m`` or
``lambda_mforce`` is an error. With neither new keyword present, the original
Cartesian objective and output columns are unchanged.

For residual :math:`\delta\boldsymbol M_i=\boldsymbol M_i-\boldsymbol M_i^{ref}`
and the input spin direction :math:`\hat{\boldsymbol s}_i`, define

.. math::

   R_\parallel &= \sqrt{\frac{1}{N_{mag}}\sum_i
     (\delta\boldsymbol M_i\cdot\hat{\boldsymbol s}_i)^2},\\
   R_\perp &= \sqrt{\frac{1}{2N_{mag}}\sum_i
     |\delta\boldsymbol M_i-(\delta\boldsymbol M_i\cdot\hat{\boldsymbol s}_i)
       \hat{\boldsymbol s}_i|^2},\\
   L_{mag} &= \lambda_{m,\parallel}R_\parallel+
     \lambda_{m,\perp}R_\perp.

Both reductions use the same labeled, active atoms as ``lambda_m``; the
normalization counts one parallel and two perpendicular degrees of freedom.
At exactly zero spin, squared residuals are assigned isotropically: one third
to the parallel channel and two thirds to the perpendicular channel. Every
nonzero spin defines a direction, without a small-moment cutoff. During
optimization, the existing squared configuration weights multiply each squared
residual; reported RMSEs are unweighted.

Native SNES minimizes a weighted sum of **RMSEs**. TorchNEP's independent
coefficients multiply **MSEs**, so its ``(1/3, 2/3)`` vector-loss mapping is not
an equivalent mapping here. In general no pair of constant coefficients makes
the sum of these two RMSEs equal to Cartesian RMSE. Use ``lambda_m`` when exact
legacy behavior is required. The raw metrics obey
:math:`3 R_M^2=R_\parallel^2+2R_\perp^2`.

``lambda_tau`` remains an independent torque term. Torque weights the
perpendicular residual by spin magnitude and is not the same as ``R_perp``.
The magnetic force itself, model format, prediction outputs, and MD dynamics
are unchanged. Native SNES has no TorchNEP second-stage lambda keywords.

In split mode, ``loss.out`` adds four columns after ``rmse_tau_test`` and before
any energy-difference columns: ``rmse_m_parallel_train``, ``rmse_m_perp_train``,
``rmse_m_parallel_test``, and ``rmse_m_perp_test``. Read the ``# columns`` header;
the Cartesian and torque diagnostics are retained. Training metrics describe
the reported batch, as for the existing loss columns. Test metrics are zero
when no test set is supplied. On restart, the objective is taken from
``nep.in``; the model/restart files do not store these loss coefficients.
