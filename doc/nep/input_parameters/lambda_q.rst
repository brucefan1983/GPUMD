.. _kw_lambda_q:
.. index::
   single: lambda_q (keyword in nep.in)

:attr:`lambda_q`
================

This keyword sets the weight :math:`\lambda_q` of the loss term associated with the **total charge** (the charge for a whole structure) in the :ref:`loss function <nep_loss_function>`.
The syntax is::

  lambda_q <weight>

Here, :attr:`<weight>` represents :math:`\lambda_q`, which must satisfy :math:`\lambda_q \geq 0` and defaults to :math:`\lambda_q = 0.1`.

This keyword is only relevant for the qNEP models.
