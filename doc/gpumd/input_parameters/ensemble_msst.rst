.. _msst:
.. _kw_ensemble_msst:
.. index::
   single: msst (keyword in run.in)
   single: MSST integrator

:attr:`ensemble` (MSST)
=======================

This keyword is used to set up a multi-scale shock technique (:term:`MSST`) integrator.
Please check [Reed2003]_ for more details.

Syntax
------

The parameters can be specified as follows::

    ensemble msst <direction> <shock_velocity> qmass <qmass_value> mu <mu_value> tscale <tscale_value> p0 <p0> v0 <v0> e0 <e0>

- :attr:`<direction>`: The direction of the shock wave. It can be ``x``, ``y`` or ``z``.
- :attr:`<shock_velocity>`: The shock velocity of the shock wave in km/s.
- :attr:`<qmass_value>`: The mass of the simulation cell.
  It affects the compression speed.
  Its unit is :math:`\mathrm{amu}^2/\mathrm{Å}^4`.
- :attr:`<mu_value>`: The artificial viscosity.
  It improves convergence.
  Its unit is :math:`\sqrt{\mathrm{amu} \times \mathrm{eV}}/\mathrm{Å}^2`.
- :attr:`<tscale_value>`: The ratio of kinetic energy that turns into cell kinetic energy.
  This helps speed up the simulation.
  This keyword is optional, and the default value is ``0``.

If keywords :attr:`<p0>`, :attr:`<v0>` or :attr:`<e0>` are not supplied, these quantities will be calculated on the first step.
In most cases, you don't need to specify these quantities.

Example
--------

.. code-block:: rst

    ensemble msst x 15 qmass 10000 mu 10

This command performs an :term:`MSST` simulation with a shock wave velocity of 15 km/s in the x direction.
