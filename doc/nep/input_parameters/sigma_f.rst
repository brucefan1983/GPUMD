.. _kw_sigma_f:
.. index::
   single: sigma_f (keyword in nep.in)

:attr:`sigma_f`
================

This keyword sets the standard deviation :math:`\sigma_\mathrm{f}` used to normalize the force term of the :ref:`loss function <nep_loss_function>` when :ref:`loss_mode <kw_loss_mode>` is set to ``1``.
The syntax is::

  sigma_f <value>

Here, :attr:`<value>` represents :math:`\sigma_\mathrm{f}` in units of eV/Å, which must satisfy :math:`\sigma_\mathrm{f} > 0` and defaults to :math:`\sigma_\mathrm{f} = 0.01`.
This keyword can only be set when :attr:`loss_mode` is set to ``1``.
