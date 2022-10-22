.. _kw_lambda_e:
.. index::
   single: lambda_e (keyword in nep.in)

:attr:`lambda_e`
================

This keyword sets the weight :math:`\lambda_e` of the loss term associated with the **energy** in the :ref:`loss function <nep_loss_function>`.
The syntax is::

  lambda_e <weight>

Here, :attr:`<weight>` represents :math:`\lambda_e`, which must satisfy :math:`\lambda_e \geq 0` and defaults to :math:`\lambda_e = 1.0`.
