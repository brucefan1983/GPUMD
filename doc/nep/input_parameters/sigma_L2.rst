.. _kw_sigma_L2:
.. index::
   single: sigma_L2 (keyword in nep.in)

:attr:`sigma_L2`
================

This keyword sets the scale :math:`\sigma_{L2}` used to normalize the :math:`\mathcal{L}_2` regularization term of the :ref:`loss function <nep_loss_function>` when :ref:`loss_mode <kw_loss_mode>` is set to ``1``.
The syntax is::

  sigma_L2 <value>

Here, :attr:`<value>` represents :math:`\sigma_{L2}`, which must satisfy :math:`\sigma_{L2} \geq 0` and defaults to :math:`\sigma_{L2} = 1.0`.
This keyword can only be set when :attr:`loss_mode` is set to ``1``.
