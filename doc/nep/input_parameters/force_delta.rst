.. _kw_force_delta:
.. index::
   single: force_delta (keyword in nep.in)

:attr:`force_delta`
===================

This keyword can be used to bias the loss function to put more emphasis on obtaining accurate predictions for smaller forces.
The syntax is::

  force_delta <delta>

where :attr:`<delta>` sets the parameter :math:`\delta`, which must satisfy :math:`\delta \geq 0` eV/A and defaults to :math:`\delta = 0` eV/A (i.e., no bias).

When :math:`\delta = 0` eV/A, the :ref:`loss term associated with the forces <nep_loss_function>` is proportional to the :term:`RMSE` of the forces:

.. math::
   
   \sqrt{\frac{1}{3N}\sum_{i=1}^{N}\left(\vec{F}_i^\mathrm{NEP} - \vec{F}_i^\mathrm{tar}\right)^2}

When :math:`\delta > 0` eV/Å, this expression is modified to read:

.. math::
   
   \sqrt{\frac{1}{3N}\sum_{i=1}^{N}\left(\vec{F}_i^\mathrm{NEP} - \vec{F}_i^\mathrm{tar}\right)^2 \frac{1}{1+\|\vec{F}_i^\mathrm{tar}\| / \delta} }

In this case, a smaller :math:`\delta` implies a larger weight on smaller forces.
