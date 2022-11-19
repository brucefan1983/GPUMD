.. _kw_lambda_shear:
.. index::
   single: lambda_shear (keyword in nep.in)

:attr:`lambda_shear`
================

This keyword sets the extra weight :math:`\lambda_s` of the loss term associated with the **shear virials** in the :ref:`loss function <nep_loss_function>`.
The syntax is::

  lambda_shear <weight>

Here, :attr:`<weight>` represents :math:`\lambda_s`, which must satisfy :math:`\lambda_s \geq 0` and defaults to :math:`\lambda_s = 1`.
The weight for the shear virials is thus :math:`\lambda_v \lambda_s`, where :math:`\lambda_v` is set by the :ref:`lambda_v keyword <kw_lambda_v>`.
We have tested that a vlaue of :math:`\lambda_s=5` is sometimes helpful to make the prediction of shear modulus more accurate.
