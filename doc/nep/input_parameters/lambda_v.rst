.. _kw_lambda_v:
.. index::
   single: lambda_v (keyword in nep.in)

:attr:`lambda_v`
================

This keyword sets the weight :math:`\lambda_v` of the loss term associated with the **virials** in the :ref:`loss function <nep_loss_function>`.
The syntax is::

  lambda_v <weight>

Here, :attr:`<weight>` represents :math:`\lambda_v`, which must satisfy :math:`\lambda_v \geq 0` and defaults to :math:`\lambda_v = 0.1`.
