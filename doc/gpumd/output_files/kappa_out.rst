.. _kappa_out:
.. index::
   single: kappa.out (output file)

``kappa.out``
=============

This file contains thermal-conductivity tensor components estimated using the homogeneous nonequilibrium molecular dynamics (:term:`HNEMD`) method.
It is generated when invoking the :ref:`compute_hnemd keyword <kw_compute_hnemd>`.
Each row is calculated from the heat current averaged over the most recent :attr:`output_interval` MD steps.
The running average can be obtained by cumulatively averaging these rows.

Results are appended to the file.

File format
-----------
If the driving force is in the positive :math:`\mu` (:math:`\mu` can be :math:`x`, :math:`y`, or :math:`z`) direction, this file reads:

* column 1: :math:`\kappa_{x\mu}^{\text{in}}(t)` (in units of W/mK)
* column 2: :math:`\kappa_{x\mu}^{\text{out}}(t)` (in units of W/mK)
* column 3: :math:`\kappa_{y\mu}^{\text{in}}(t)` (in units of W/mK)
* column 4: :math:`\kappa_{y\mu}^{\text{out}}(t)` (in units of W/mK)
* column 5: :math:`\kappa_{z\mu}^{\text{tot}}(t)` (in units of W/mK)

Both :math:`\kappa_{x\mu}(t)` and :math:`\kappa_{y\mu}(t)` have been decomposed into contributions from in-plane (hence superscript in) and out-of-plane (hence superscript out) vibrational modes, as described in [Fan2019]_.
This decomposition is useful for 2D (or layered) materials but is not necessary for 3D materials.
For 3D materials, one can sum up some columns to get the conventional data.
That is:

.. math::

   \kappa_{x\mu}^{\text{tot}}(t) &= \kappa_{x\mu}^{\text{in}}(t) + \kappa_{x\mu}^{\text{out}}(t) \\
   \kappa_{y\mu}^{\text{tot}}(t) &= \kappa_{y\mu}^{\text{in}}(t) + \kappa_{y\mu}^{\text{out}}(t).

Only the potential part of the heat current has been considered.
For systems in which convective heat transport is important, use the :ref:`compute_hnemdec keyword <kw_compute_hnemdec>`, which writes :ref:`onsager.out <onsager_out>`.
