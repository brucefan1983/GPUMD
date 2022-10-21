.. _hac_out:
.. index::
   single: hac.out (output file)

``hac.out``
===========

This file contains the heat current auto-correlation (:term:`HAC`) function and the running thermal conductivity (:term:`RTC`) from the the :ref:`EMD method for heat transport <green_kubo_method>` method.
It is produced when invoking the :ref:`compute_hac keyword <kw_compute_hac>` in the :ref:`run.in input file <run_in>`.

File format
-----------
This file reads

* column 1: correlation time (in units of ps)
* column 2: :math:`\langle J_x^{\text{in}}(0)J_x^{\text{tot}}(t)\rangle` (in units of eV\ :math:`^3`/amu)
* column 3: :math:`\langle J_x^{\text{out}}(0)J_x^{\text{tot}}(t)\rangle` (in units of eV\ :math:`^3`/amu)
* column 4: :math:`\langle J_y^{\text{in}}(0)J_y^{\text{tot}}(t)\rangle` (in units of eV\ :math:`^3`/amu)
* column 5: :math:`\langle J_y^{\text{out}}(0)J_y^{\text{tot}}(t)\rangle` (in units of eV\ :math:`^3`/amu)
* column 6: :math:`\langle J_z^{\text{tot}}(0)J_z^{\text{tot}}(t)\rangle` (in units of eV\ :math:`^3`/amu)
* column 7: :math:`\kappa_x^{\text{in}}(t)` (in units of W/mK)
* column 8: :math:`\kappa_x^{\text{out}}(t)` (in units of W/mK)
* column 9: :math:`\kappa_y^{\text{in}}(t)` (in units of W/mK)
* column 10: :math:`\kappa_y^{\text{out}}(t)` (in units of W/mK)
* column 11: :math:`\kappa_z^{\text{tot}}(t)` (in units of W/mK)

Note that the :term:`HAC` and the :term:`RTC` are decomposed as described in [Fan2017]_.
This decomposition is useful for 2D materials but not necessary for 3D materials.
For 3D materials, one can sum up some columns to get the conventional data.
For example:

.. math::

   \langle J_x^{\text{tot}}(0)J_x^{\text{tot}}(t) \rangle
   = \langle J_x^{\text{in}}(0)J_x^{\text{tot}}(t) \rangle
   + \langle J_x^{\text{out}}(0)J_x^{\text{tot}}(t) \rangle

and

.. math::
   
   \kappa_x^{\text{tot}}(t) = \kappa_x^{\text{in}}(t) + \kappa_x^{\text{out}}(t).

Note that the cross term introduced in [Fan2017]_ has been evenly attributed to the in-plane and out-of-plane components.
This has been justified in [Fan2019]_.

Only the potential part of the heat current is included.
If the convective part of the heat current is important in your system, you can use the :ref:`compute keyword <kw_compute>` to calculate and output the heat current data and post-process it by yourself.
