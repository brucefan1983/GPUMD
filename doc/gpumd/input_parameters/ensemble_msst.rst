.. _kw_ensemble_msst:

:attr:`ensemble` (MSST)
=======================

This keyword is used to set up a Multi-Scale Shock Technique (MSST) integrator. Please check [Reed2003]_ for more details.

Syntax
------

The parameters can be specified as follows::

    ensemble msst <direction> <shock_velocity> qmass <qmass_value> mu <mu_value> tscale <tscale_value>

- :attr:`<direction>`: The direction of the shock wave. It can be ``x``, ``y``, or ``z``.
- :attr:`<shock_velocity>`: The shock velocity of the shock wave in km/s.
- :attr:`<qmass_value>`: The mass of the simulation cell. It affects the compression speed. Its unit is :math:`\frac{amu^2}{Å^4}`.
- :attr:`<mu_value>`: The artificial viscosity. It improves convergence. Its unit is :math:`\frac{\sqrt{amu \times eV}}{Å^2}`.
- :attr:`<tscale>`: The ratio of kinetic energy that turns into cell kinetic energy. This helps speed up the simulation. This keyword is optional, and the default value is ``0``.

Example
--------

.. code-block:: rst

    ensemble msst x 15 qmass 10000 mu 10

This command performs an MSST simulation with a shock wave velocity of 15 km/s in the x direction.