.. _viscosity_out:
.. index::
   single: viscosity.out (output file)

``viscosity.out``
=================

This file contains the stress auto-correlation function and the running viscosity from the Green-Kubo method.
It is produced when invoking the :ref:`compute_viscosity keyword <kw_compute_viscosity>` in the :ref:`run.in input file <run_in>`.

File format
-----------
This file reads

* column  1: correlation time (in units of ps)
* column  2: :math:`\langle S_{xx}(0)S_{xx}(t)\rangle` (in units of eV\ :math:`^2`)
* column  3: :math:`\langle S_{yy}(0)S_{yy}(t)\rangle` (in units of eV\ :math:`^2`)
* column  4: :math:`\langle S_{zz}(0)S_{zz}(t)\rangle` (in units of eV\ :math:`^2`)
* column  5: :math:`\langle S_{xy}(0)S_{xy}(t)\rangle` (in units of eV\ :math:`^2`)
* column  6: :math:`\langle S_{xz}(0)S_{xz}(t)\rangle` (in units of eV\ :math:`^2`)
* column  7: :math:`\langle S_{yz}(0)S_{yz}(t)\rangle` (in units of eV\ :math:`^2`)
* column  8: :math:`\langle S_{yx}(0)S_{yx}(t)\rangle` (in units of eV\ :math:`^2`)
* column  9: :math:`\langle S_{zx}(0)S_{zx}(t)\rangle` (in units of eV\ :math:`^2`)
* column 10: :math:`\langle S_{zy}(0)S_{zy}(t)\rangle` (in units of eV\ :math:`^2`)
* column 11: :math:`\frac{1}{k_{\rm B}TV}\int_0^t\langle S_{xx}(0)S_{xx}(t')\rangle dt'` (in units of mPa s)
* column 12: :math:`\frac{1}{k_{\rm B}TV}\int_0^t\langle S_{yy}(0)S_{yy}(t')\rangle dt'` (in units of mPa s)
* column 13: :math:`\frac{1}{k_{\rm B}TV}\int_0^t\langle S_{zz}(0)S_{zz}(t')\rangle dt'` (in units of mPa s)
* column 14: :math:`\frac{1}{k_{\rm B}TV}\int_0^t\langle S_{xy}(0)S_{xy}(t')\rangle dt'` (in units of mPa s)
* column 15: :math:`\frac{1}{k_{\rm B}TV}\int_0^t\langle S_{xz}(0)S_{xz}(t')\rangle dt'` (in units of mPa s)
* column 16: :math:`\frac{1}{k_{\rm B}TV}\int_0^t\langle S_{yz}(0)S_{yz}(t')\rangle dt'` (in units of mPa s)
* column 17: :math:`\frac{1}{k_{\rm B}TV}\int_0^t\langle S_{yx}(0)S_{yx}(t')\rangle dt'` (in units of mPa s)
* column 18: :math:`\frac{1}{k_{\rm B}TV}\int_0^t\langle S_{zx}(0)S_{zx}(t')\rangle dt'` (in units of mPa s)
* column 19: :math:`\frac{1}{k_{\rm B}TV}\int_0^t\langle S_{zy}(0)S_{zy}(t')\rangle dt'` (in units of mPa s)
