.. _kw_lambda_1:
.. index::
   single: lambda_1 (keyword in nep.in)

:attr:`lambda_1`
================

This keyword sets the weight :math:`\lambda_1` of the :math:`\mathcal{L}_1`-norm regularization term in the :ref:`loss function <nep_loss_function>`.
The syntax is::

  lambda_1 <weight>

Here, :attr:`<weight>` represents :math:`\lambda_1`, which must satisfy :math:`\lambda_1 \geq 0` and defaults to :math:`\lambda_1 = 0.05`.
It is often beneficial for model stability to use a strong regularization with :math:`\lambda_1 = 0.1`.
