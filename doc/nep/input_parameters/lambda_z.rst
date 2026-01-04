.. _kw_lambda_z:
.. index::
   single: lambda_z (keyword in nep.in)

:attr:`lambda_z`
================

This keyword sets the weight :math:`\lambda_z` of the loss term associated with the **Born effective charge** in the :ref:`loss function <nep_loss_function>`.
The syntax is::

  lambda_z <weight>

Here, :attr:`<weight>` represents :math:`\lambda_z`, which must satisfy :math:`\lambda_z \geq 0` and defaults to :math:`\lambda_z = 0.5`.

This keyword is only relevant for the qNEP models.
