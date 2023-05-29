.. _onsager_out:
.. index::
   single: onsager.out (output file)

``onsager.out``
===============

This file contains some components of the running onsager coefficients tensor from the homogeneous non-equilibrium molecular dynamics Evans-Cummings algorithm (:term:`HNEMDEC`) method.
It is generated when invoking the :ref:`compute_hnemdec keyword <kw_compute_hnemdec>`.

File format
-----------
If the driving force is in the :math:`\mu` (:math:`\mu` can be :math:`x`, :math:`y`, or :math:`z`) direction and the dissipative flux is set as heat flux, for a system with :math:`M` elements, this file reads:
  
* column 1: :math:`L_{x \mu}^{\text{qq}}(t)` (in units of :math:`W/mK`)
* column 2: :math:`L_{y \mu}^{\text{qq}}(t)` (in units of :math:`W/mK`)
* column 3: :math:`L_{z \mu}^{\text{qq}}(t)` (in units of :math:`W/mK`)
* column 4: :math:`L_{x \mu}^{\text{1q}}(t)` (in units of :math:`10^{-6} kg/smK`)
* column 5: :math:`L_{y \mu}^{\text{1q}}(t)` (in units of :math:`10^{-6} kg/s/m/K`)
* column 6: :math:`L_{z \mu}^{\text{1q}}(t)` (in units of :math:`10^{-6} kg/s/m/K`)
* ...
* column 3M+1: :math:`L_{x \mu}^{\text{Mq}}(t)` (in units of :math:`10^{-6} kg/smK`)
* column 3M+2: :math:`L_{y \mu}^{\text{Mq}}(t)` (in units of :math:`10^{-6} kg/smK`)
* column 3M+3: :math:`L_{z \mu}^{\text{Mq}}(t)` (in units of :math:`10^{-6} kg/smK`)


If the dissipative flux is changed to momentum flux of component :math:`\alpha`, this file reads:

* column 1: :math:`L_{x \mu}^{q\alpha}(t)` (in units of :math:`10^{-6} kg/smK`)
* column 2: :math:`L_{y \mu}^{q\alpha}(t)` (in units of :math:`10^{-6} kg/smK`)
* column 3: :math:`L_{z \mu}^{q\alpha}(t)` (in units of :math:`10^{-6} kg/smK`)
* column 4: :math:`L_{x \mu}^{1\alpha}(t)` (in units of :math:`10^{-12} kgs/m^{3}K`)
* column 5: :math:`L_{y \mu}^{1\alpha}(t)` (in units of :math:`10^{-12} kgs/m^{3}K`)
* column 6: :math:`L_{z \mu}^{1\alpha}(t)` (in units of :math:`10^{-12} kgs/m^{3}K`)
* ...  
* column :math:`3M+1`: :math:`L_{x \mu}^{M\alpha}(t)` (in units of :math:`10^{-12} kgs/m^{3}K`)
* column :math:`3M+2`: :math:`L_{y \mu}^{M\alpha}(t)` (in units of :math:`10^{-12} kgs/m^{3}K`)
* column :math:`3M+3`: :math:`L_{z \mu}^{M\alpha}(t)` (in units of :math:`10^{-12} kgs/m^{3}K`)

Both the potential part and the kinetic part of the heat current have been considered.
One can obtain the onsager coefficent :math:`\Lambda_{i j}^{ml}` by:

.. math::

   \Lambda_{i j}^{ml}=T^{2}L_{i j}^{ml}

The thermal conductivity can be derived from the onsager matrix :math:`\Lambda`, that is:

.. math::

   \kappa=\frac{1}{T^{2}(\Lambda^{-1})_{00}}
