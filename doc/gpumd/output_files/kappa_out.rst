.. _kappa_out:
.. index::
   single: kappa.out (output file)

``kappa.out``
=============

This file contains some components of the running thermal conductivity (:term:`RTC`) tensor from the homogeneous nonequilibrium molecular dynamics (:term:`HNEMD`) method.
It is generated when invoking the :ref:`compute_hnemd keyword <kw_compute_hnemd>`.

File format
-----------
If the driving force is in the :math:`\mu` (:math:`\mu` can be :math:`x`, :math:`y`, or :math:`z`) direction, this file reads:
  
* column 1: :math:`\kappa_{\mu x}^{\text{in}}(t)` (in units of W/mK)
* column 2: :math:`\kappa_{\mu x}^{\text{out}}(t)` (in units of W/mK)
* column 3: :math:`\kappa_{\mu y}^{\text{in}}(t)` (in units of W/mK)
* column 4: :math:`\kappa_{\mu y}^{\text{out}}(t)` (in units of W/mK)
* column 5: :math:`\kappa_{\mu z}^{\text{tot}}(t)` (in units of W/mK)

Both :math:`\kappa_{\mu x}(t)` and :math:`\kappa_{\mu y}(t)` have been decomposed into comtributions from in-plane (hence superscript in) and out-of-plane (hence superscript out) vibrational modes, as described in [Fan2019]_.
This decomposition is useful for 2D (or layered) materials but is not necessary for 3D materials.
For 3D materials, one can sum up some columns to get the conventional data.
That is:

.. math::
   
   \kappa_{\mu x}^{\text{tot}}(t) &= \kappa_{\mu x}^{\text{in}}(t) + \kappa_{\mu x}^{\text{out}}(t) \\
   \kappa_{\mu y}^{\text{tot}}(t) &= \kappa_{\mu y}^{\text{in}}(t) + \kappa_{\mu y}^{\text{out}}(t).

Only the potential part of the heat current has been considered.
To simulation systems in which the convective heat current is important, one would have to modify the source code.
