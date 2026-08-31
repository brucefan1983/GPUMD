.. _kw_loss_mode:
.. index::
   single: loss_mode (keyword in nep.in)

:attr:`loss_mode`
==================

This keyword selects the form of the :ref:`loss function <nep_loss_function>` used during training.
The syntax is::

  loss_mode <mode>

Here, :attr:`<mode>` can take the following values:

* ``0``: the default RMSE-based loss function, where the energy, force, and virial terms are the respective :term:`RMSE` values between the :term:`NEP` predictions and the target values, and the :math:`\mathcal{L}_1`/:math:`\mathcal{L}_2` regularization terms are unnormalized.
* ``1``: a normalized MSE-based loss function, where the energy, force, and virial (stress) terms are instead the mean squared errors divided by the squared standard deviations :math:`\sigma_\mathrm{e}^2`, :math:`\sigma_\mathrm{f}^2`, and :math:`\sigma_\mathrm{s}^2` (see :ref:`sigma_e <kw_sigma_e>`, :ref:`sigma_f <kw_sigma_f>`, and :ref:`sigma_s <kw_sigma_s>`), and the :math:`\mathcal{L}_1`/:math:`\mathcal{L}_2` regularization terms are similarly normalized by :math:`\sigma_{L1}` and :math:`\sigma_{L2}^2` (see :ref:`sigma_L1 <kw_sigma_L1>` and :ref:`sigma_L2 <kw_sigma_L2>`).

The default is :math:`\mathtt{loss\_mode} = 0`.
The :attr:`sigma_e`, :attr:`sigma_f`, :attr:`sigma_s`, :attr:`sigma_L1`, and :attr:`sigma_L2` keywords can only be set when :attr:`loss_mode` is set to ``1``.

When :attr:`loss_mode` is set to ``1``, the :ref:`loss function <nep_loss_function>` reads

.. math::

   L(\boldsymbol{z})
   &= \lambda_\mathrm{e} \frac{1}{N_\mathrm{str}\sigma_\mathrm{e}^2}\sum_{n=1}^{N_\mathrm{str}} \left( U^\mathrm{NEP}(n,\boldsymbol{z}) - U^\mathrm{tar}(n)\right)^2 \nonumber \\
   &+  \lambda_\mathrm{f} \frac{1}{3N\sigma_\mathrm{f}^2}
   \sum_{i=1}^{N} \left( \boldsymbol{F}_i^\mathrm{NEP}(\boldsymbol{z}) - \boldsymbol{F}_i^\mathrm{tar}\right)^2 \nonumber \\
   &+  \lambda_\mathrm{v} \frac{1}{6N_\mathrm{str}\sigma_\mathrm{s}^2}
   \sum_{n=1}^{N_\mathrm{str}} \sum_{\mu\nu} \left( P_{\mu\nu}^\mathrm{NEP}(n,\boldsymbol{z}) - P_{\mu\nu}^\mathrm{tar}(n)\right)^2 \nonumber \\
   &+  \lambda_1 \frac{1}{\sigma_{L1} N_\mathrm{par}} \sum_{n=1}^{N_\mathrm{par}} |z_n| \nonumber \\
   &+  \lambda_2 \frac{1}{\sigma_{L2}^2 N_\mathrm{par}} \sum_{n=1}^{N_\mathrm{par}} z_n^2,

where :math:`P_{\mu\nu}^\mathrm{NEP}(n,\boldsymbol{z})` and :math:`P_{\mu\nu}^\mathrm{tar}(n)` are the predicted and target stress tensor components (in GPa) for the :math:`n^\mathrm{th}` structure, and :math:`\sigma_\mathrm{e}`, :math:`\sigma_\mathrm{f}`, :math:`\sigma_\mathrm{s}`, :math:`\sigma_{L1}`, and :math:`\sigma_{L2}` are set via the :ref:`sigma_e <kw_sigma_e>`, :ref:`sigma_f <kw_sigma_f>`, :ref:`sigma_s <kw_sigma_s>`, :ref:`sigma_L1 <kw_sigma_L1>`, and :ref:`sigma_L2 <kw_sigma_L2>` keywords, respectively.
The other symbols are as defined for the default loss function above.
Unlike the default case, the energy, force, and stress terms here are mean squared errors rather than :term:`RMSE` values, and the :math:`\mathcal{L}_1`/:math:`\mathcal{L}_2` regularization terms are normalized by :math:`\sigma_{L1}` and :math:`\sigma_{L2}^2`, respectively, rather than left unnormalized.
