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

In practice, one can first estimate the weighted sum of the loss terms for the target quantities upon convergence and then 
set :math:`\lambda_1` to that value. For example, if one estimates the weighted sum of the loss terms for energy, force, 
and virial to be :math:`0.01 + 0.1 + 0.01`, then it is reasonable to set :math:`\lambda_1 = 0.12`.
