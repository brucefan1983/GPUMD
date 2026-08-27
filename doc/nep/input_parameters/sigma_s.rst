.. _kw_sigma_s:
.. index::
   single: sigma_s (keyword in nep.in)

:attr:`sigma_s`
================

This keyword sets the standard deviation :math:`\sigma_\mathrm{s}` used to normalize the stress term of the :ref:`loss function <nep_loss_function>` when :ref:`loss_mode <kw_loss_mode>` is set to ``1``.
The syntax is::

  sigma_s <value>

Here, :attr:`<value>` represents :math:`\sigma_\mathrm{s}` in units of GPa, which must satisfy :math:`\sigma_\mathrm{s} > 0` and defaults to :math:`\sigma_\mathrm{s} = 0.1`.
This keyword can only be set when :attr:`loss_mode` is set to ``1``.
