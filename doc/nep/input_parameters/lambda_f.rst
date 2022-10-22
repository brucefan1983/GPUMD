.. _kw_lambda_f:
.. index::
   single: lambda_f (keyword in nep.in)

:attr:`lambda_f`
================

This keyword sets the weight :math:`\lambda_f` of the loss term associated with the **forces** in the :ref:`loss function <nep_loss_function>`.
The syntax is::

  lambda_f <weight>

Here, :attr:`<weight>` represents :math:`\lambda_f`, which must satisfy :math:`\lambda_f \geq 0` and defaults to :math:`\lambda_f = 1.0`.
