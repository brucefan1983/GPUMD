.. _kw_lambda_spin_response:
.. index::
   single: lambda_spin_response (keyword in nep.in)

:attr:`lambda_spin_response`
=============================

This keyword sets the weight of grouped magnetic-response loss::

  lambda_spin_response <weight>

The weight must be finite and non-negative and defaults to 0. A positive value
requires ``spin_version 1`` and a valid :ref:`response.xyz <response_xyz>` in
the working directory. With weight 0, the file is not opened. See that file's
page for required labels, group metadata, and data-splitting rules.

All complete groups are evaluated for every candidate in every SNES generation
after initialization. Ordinary :ref:`batch <kw_batch>` settings do not split
response groups. Response frames contribute only this loss and reuse the
energy reference and scaler obtained from ordinary training data. Test
reporting and model selection are unchanged; no separate response validation
loss is added.

The loss compares predicted and target generalized magnetic responses along
each path. Each group has equal weight. Its uncentered target-response RMS,
floored at :math:`10^{-6}`, normalizes a centered-shape Huber term and a
group-mean Huber term; the latter has weight 0.25. Coordinates do not add a
separate reliability weight. This objective supplements per-component
magnetic-force RMSE when relative response along a controlled rotation path
matters. See :ref:`nep_spin_response` for the response and tangent definitions.
