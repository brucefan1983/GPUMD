.. _kw_lambda_e:
.. index::
   single: lambda_e (keyword in nep.in)

:attr:`lambda_e`
================

This keyword sets the weight :math:`\lambda_e` of the loss term associated with the **energy** in the :ref:`loss function <nep_loss_function>`.
The syntax is::

  lambda_e <weight>

Here, :attr:`<weight>` represents :math:`\lambda_e`, which must satisfy :math:`\lambda_e \geq 0` and defaults to :math:`\lambda_e = 1.0`.

It might be beneficial to use a two-step training scheme, where :math:`\lambda_e` takes a small value (such as 0.1) in the first training and a large value (such as 10) in the second (restarting from the first).
During the second training, the energy error might decrease appreciably, while the force and virial errors are not much affected.
