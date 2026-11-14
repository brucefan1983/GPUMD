.. _kw_force_delta:
.. index::
   single: force_delta (keyword in nep.in)

:attr:`force_delta`
===================

This keyword can be used to bias the loss function to put more emphasis on obtaining accurate predictions for smaller forces.

Syntax
------

::

  force_delta <delta>

Parameters
----------

:attr:`<delta>` sets the parameter :math:`\delta`, which must satisfy :math:`\delta \geq 0` eV/A and defaults to :math:`\delta = 0` eV/A (i.e., no bias).

Output
------

None.

Examples
--------

::

  force_delta 1.0

Notes
-----

When :math:`\delta = 0` eV/A, the :ref:`loss term associated with the forces <nep_loss_function>` is proportional to the :term:`RMSE` of the forces:

.. math::

   \sqrt{\frac{1}{3N}\sum_{i=1}^{N}\left(\boldsymbol{F}_i^\mathrm{NEP} - \boldsymbol{F}_i^\mathrm{tar}\right)^2}

When :math:`\delta > 0` eV/Å, this expression is modified to read:

.. math::

   \sqrt{\frac{1}{3N}\sum_{i=1}^{N}\left(\boldsymbol{F}_i^\mathrm{NEP} - \boldsymbol{F}_i^\mathrm{tar}\right)^2 \frac{1}{1+\|\boldsymbol{F}_i^\mathrm{tar}\| / \delta} }

In this case, a smaller :math:`\delta` implies a larger weight on smaller forces.
