.. _kw_lambda_1:
.. index::
   single: lambda_1 (keyword in nep.in)

:attr:`lambda_1`
================

This keyword sets the weight :math:`\lambda_1` of the :math:`\mathcal{L}_1`-norm regularization term in the :ref:`loss function <nep_loss_function>`.
The syntax is::

  lambda_1 <weight>

Here, :attr:`<weight>` represents :math:`\lambda_1`, which can be set to any non-negative values. 
The default value is 0, which means that :math:`\mathcal{L}_1` regularization is not used by default.
