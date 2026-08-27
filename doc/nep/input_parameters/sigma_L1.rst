.. _kw_sigma_L1:
.. index::
   single: sigma_L1 (keyword in nep.in)

:attr:`sigma_L1`
================

This keyword sets the scale :math:`\sigma_{L1}` used to normalize the :math:`\mathcal{L}_1` regularization term of the :ref:`loss function <nep_loss_function>` when :ref:`loss_mode <kw_loss_mode>` is set to ``1``.
The syntax is::

  sigma_L1 <value>

Here, :attr:`<value>` represents :math:`\sigma_{L1}`, which must satisfy :math:`\sigma_{L1} \geq 0` and defaults to :math:`\sigma_{L1} = 0.001`.
This keyword can only be set when :attr:`loss_mode` is set to ``1``.
