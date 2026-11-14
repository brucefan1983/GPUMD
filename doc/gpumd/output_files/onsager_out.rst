.. _onsager_out:
.. index::
   single: onsager.out (output file)

``onsager.out``
===============

This file contains block averages of components of the Onsager coefficient tensors from the homogeneous non-equilibrium molecular dynamics Evans-Cummings algorithm (:term:`HNEMDEC`) method.
It is generated when invoking the :ref:`compute_hnemdec keyword <kw_compute_hnemdec>`.
Each row averages the preceding :attr:`output_interval` MD steps, rather than all steps since the start of the run.
Rows are appended to the file, with :math:`3(M+1)` columns for :math:`M` elements and no time column.

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
One can obtain the onsager coefficient :math:`\Lambda_{i j}^{ml}` by:

.. math::

   \Lambda_{i j}^{ml}=T^{2}L_{i j}^{ml}

Obtaining the thermal conductivity
----------------------------------

For each driving direction, use separate calculations with thermal driving (:attr:`driving_force` = 0) and diffusive driving for :math:`M-1` selected components.
Use the same temperature, composition, and density, with a sufficiently small positive field along one Cartesian axis per calculation.
Average the output blocks after equilibration, and save the output from each calculation separately.

At zero total momentum, the component momentum currents obey :math:`\sum_{\alpha=1}^{M}\boldsymbol{J}_{\alpha}=0`.
Keep the heat current and the same :math:`M-1` independent momentum currents when assembling rows and driving-force columns of the Onsager matrix.
Including all :math:`M` component momentum currents gives a linearly dependent matrix.

Convert the printed values to SI units before assembling the matrix: multiply heat--momentum and momentum--heat coefficients by :math:`10^{-6}`, and momentum--momentum coefficients by :math:`10^{-12}`.
Heat--heat coefficients already have units of W/(m K).
The response-current index specifies the row and the driven-current index specifies the column.

Writing the matrix of coefficients :math:`L=\Lambda/T^2` in heat (:math:`q`) and independent momentum (:math:`d`) blocks, the thermal conductivity at zero component mass currents is

.. math::

   \boldsymbol{\kappa}=L^{qq}-L^{qd}(L^{dd})^{-1}L^{dq}.

For general anisotropic transport, these blocks include the Cartesian components obtained from calculations driven along all three axes.
For an isotropic system or a Cartesian direction decoupled from the other directions, the calculation reduces to an :math:`M\times M` matrix for that direction.

For example, consider a binary system driven along :math:`x`, retaining the momentum current of the first component.
Use :attr:`driving_force` = 0 and 1 in separate calculations, and save their outputs as ``onsager_heat.out`` and ``onsager_diffusion.out``.
The longitudinal conductivity can then be calculated as follows (the example discards the first 100 output blocks; choose this number from the equilibration behavior):

.. code-block:: python

   import numpy as np

   n_skip = 100
   thermal = np.loadtxt("onsager_heat.out", ndmin=2)[n_skip:].mean(axis=0)
   diffusive = np.loadtxt("onsager_diffusion.out", ndmin=2)[n_skip:].mean(axis=0)
   Lqq = thermal[0]
   L1q = thermal[3] * 1e-6
   Lq1 = diffusive[0] * 1e-6
   L11 = diffusive[3] * 1e-12
   kappa_xx = Lqq - Lq1 * L1q / L11

The result is in W/(m K), and no additional factor of :math:`T^2` is needed.
In this binary expression, the unit-conversion factors cancel because :math:`10^{-6}\times10^{-6}/10^{-12}=1`, so the same result can also be obtained directly from the printed values.
The `Ar-Kr HNEMDEC tutorial <https://github.com/brucefan1983/GPUMD-Tutorials/tree/main/examples/29_thermal_transport_multicomponent_HNEMDEC>`_ provides input files, output data, and a post-processing script that uses this cancellation and also plots running averages of the output blocks.
