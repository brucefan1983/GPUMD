.. _kw_lambda_2:
.. index::
   single: lambda_2 (keyword in nep.in)

:attr:`lambda_2`
================

This keyword sets the weight :math:`\lambda_2` of the :math:`\mathcal{L}_2`-norm regularization term in the :ref:`loss function <nep_loss_function>`.
The syntax is::

  lambda_2 <weight>

Here, :attr:`<weight>` represents :math:`\lambda_2`, which can be set to any non-negative values. 
The default value is :math:`\lambda_2 = \sqrt{N}/1000`, where :math:`N` is the total number of training parameters.
