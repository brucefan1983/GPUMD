.. _kw_lambda_tau:
.. index::
   single: lambda_tau (keyword in nep.in)

:attr:`lambda_tau`
===================

This keyword sets the weight of spin-torque RMSE in the loss::

  lambda_tau <weight>

The weight must be non-negative and defaults to 0. For each labeled active
atom, the predicted and target quantities are converted to

.. math::

   \boldsymbol{\tau}_i
   = \boldsymbol{s}_i\times\boldsymbol{M}_i.

The resulting torque RMSE emphasizes magnetic-force components perpendicular
to the current spin. Only frames containing ``mforce:R:3`` and atom types
enabled by :ref:`spin_type <kw_spin_type>` contribute.

The Cartesian loss :ref:`lambda_m <kw_lambda_m>` constrains all magnetic-force
components, whereas torque is insensitive to the parallel component.
Set ``lambda_m 0`` and a positive ``lambda_tau`` to supervise spin rotation
without constraining the longitudinal magnetic force. Torque RMSE includes only
labeled active spins with :math:`|\boldsymbol{s}_i|^2 > 10^{-20}\,\mu_\mathrm{B}^2`
and is normalized by two degrees of freedom per included spin. The squared
magnitude is evaluated in double precision for both the error and its count;
spins at or below the threshold are excluded from both. Squared torque errors
weight squared transverse magnetic-force errors by :math:`|\boldsymbol{s}_i|^2`.
