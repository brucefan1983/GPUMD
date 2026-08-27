.. _kw_sigma_e:
.. index::
   single: sigma_e (keyword in nep.in)

:attr:`sigma_e`
================

This keyword sets the standard deviation :math:`\sigma_\mathrm{e}` used to normalize the energy term of the :ref:`loss function <nep_loss_function>` when :ref:`loss_mode <kw_loss_mode>` is set to ``1``.
The syntax is::

  sigma_e <value>

Here, :attr:`<value>` represents :math:`\sigma_\mathrm{e}` in units of eV/atom, which must satisfy :math:`\sigma_\mathrm{e} > 0` and defaults to :math:`\sigma_\mathrm{e} = 0.001`.
This keyword can only be set when :attr:`loss_mode` is set to ``1``.
