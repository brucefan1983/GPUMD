.. _kw_ensemble_ti_liquid:

:attr:`ensemble` (TI_Liquid)
============================

This keyword is used to set up a nonequilibrium thermodynamic integration integrator. See [Leite2016]_, [Leite2019]_ and [Menon2021]_ for more details.

Syntax
------

::

    ensemble ti_liquid temp <temperature> tperiod <tau_temperature> tequil <equilibrium_time> tswitch <switch_time> press <pressure> sigmasqrd <sigmasqrd-value> p <p-value>

Parameters
----------

- :attr:`<temperature>`: The temperature of the simulation.
- :attr:`<tau_temperature>`: Thermostat coupling period in time steps; optional, with a default of ``100``.
- :attr:`<equilibrium_time>`: The number of time steps to equilibrate the system.
- :attr:`<switch_time>`: The number of time steps to vary lambda from 0 to 1.
- :attr:`<sigmasqrd>` and :attr:`<p>`: Specify the TI settings. Implemented values for :attr:`<p>` are ``1``, ``25``, ``50``, ``75``, ``100``.
- :attr:`<pressure>`: Although this is an NVT ensemble, you can assign a pressure value to help GPUMD compute the Gibbs free energy. It does not affect the simulation process.

Specify both ``tequil`` and ``tswitch``, or omit both.
``tequil`` must be a non-negative integer and ``tswitch`` a positive integer.
When omitted, they are set to ``floor(0.1 * N)`` and ``floor(0.4 * N)``, respectively, where ``N`` is the number of run steps; the resulting ``tswitch`` must still be positive.
The run must contain at least ``2 * (tequil + tswitch)`` steps.

Examples
--------

.. code-block:: text

    ensemble ti_liquid temp 1000 tswitch 4000 tequil 1000 press 0 tperiod 100 sigmasqrd 2 p 100

Each switching stage lasts 4000 time steps and each equilibration stage lasts 1000 time steps.

Output file
-----------

This command will produce a csv file. The columns are lambda, dlambda, potential energy and UF energy (eV/atom).

This command will also produce a yaml file, which contains Gibbs free energy and other information.
