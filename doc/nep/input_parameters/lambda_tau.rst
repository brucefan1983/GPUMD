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
enabled by :ref:`spin_dof_type <kw_spin_dof_type>` contribute.

The Cartesian loss :ref:`lambda_m <kw_lambda_m>` constrains all magnetic-force
components, whereas torque is insensitive to the parallel component.
Set ``lambda_m 0`` and a positive ``lambda_tau`` to supervise spin rotation
without constraining the longitudinal magnetic force. Zero spins are excluded
from torque RMSE, which is normalized by two degrees of freedom per active
nonzero spin. Squared torque errors weight squared transverse magnetic-force
errors by :math:`|\boldsymbol{s}_i|^2`.
