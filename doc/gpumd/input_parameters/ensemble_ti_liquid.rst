.. _kw_ensemble_ti_liquid:

:attr:`ensemble` (TI_Liquid)
============================

This keyword is used to set up a nonequilibrium thermodynamic integration integrator. Please check [Leite2016]_, [Leite2019]_ and [Menon2021]_ for more details.

Syntax
------

The parameters can be specified as follows::

    ensemble ti_liquid temp <temperature> tperiod <tau_temperature> tequil <equilibrium_time> tswitch <switch_time> press <pressure> sigmasqrd <sigmasqrd-value> p <p-value>

- :attr:`<temperature>`: The temperature of the simulation.
- :attr:`<tau_temperature>`: This parameter is optional, and defaults to ``100``. It determines the period of the thermostat in units of the timestep. It determines how strongly the system is coupled to the thermostat.
- :attr:`<equilibrium_time>`: The number timesteps to equilibrate the system.
- :attr:`<switch_time>`: The number timesteps to vary lambda from 0 to 1.
- :attr:`<sigmasqrd>` and :attr:`<p>`: Specify the TI settings. Implemented values for :attr:`<p>` are ``1``, ``25``, ``50``, ``75``, ``100``.
- :attr:`<pressure>`: Although this is an NVT ensemble, you can assign a pressure value to help GPUMD compute the Gibbs free energy. It does not affect the simlation process.

If you do not specify :attr:`<equilibrium_time>` and :attr:`<switch_time>`, they will be automatically set in a 1:4 ratio.

Example
-------

.. code-block:: rst

    ensemble ti_liquid temp 1000 tswitch 4000 tequil 1000 press 0 tperiod 100 sigmasqrd 2 p 100

This command switch lambda for 4000 timesteps, and equilibrate for 1000 timesteps.

Output file
-----------

This command will produce a csv file. The columns are lambda, dlambda, potential energy and UF energy (eV/atom).

This command will also produce a yaml file, which contains Gibbs free energy and other information.

Additionally, important information will be displayed on the screen during the simulation process, so it is recommended to carefully review and take note of these details.