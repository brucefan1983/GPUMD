.. _kw_lambda_2:
.. index::
   single: lambda_2 (keyword in nep.in)

:attr:`lambda_2`
================

This keyword sets the weight :math:`\lambda_2` of the :math:`\mathcal{L}_2`-norm regularization term in the :ref:`loss function <nep_loss_function>`.
The syntax is::

  lambda_2 <weight>

Here, :attr:`<weight>` represents :math:`\lambda_2`, which must satisfy :math:`\lambda_2 \geq 0` and defaults to :math:`\lambda_2 = 0.05`.
It is often beneficial for model stability to use a strong regularization with :math:`\lambda_2 = 0.1`.
