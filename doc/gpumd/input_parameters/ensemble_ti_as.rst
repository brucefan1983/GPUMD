.. _kw_ensemble_ti_as:

:attr:`ensemble` (TI_AS)
========================

This keyword is used to set up a nonequilibrium thermodynamic integration integrator with Adiabatic switching (AS) path. It calculates Gibbs free energy along an isothermal path. Please check [Cajahuaringa2022]_ for more details.

Syntax
------

The parameters can be specified as follows::

    ensemble ti_rs temp <temperature> tperiod <tau_temperature> <pressure_control> <pmin> <pmax> pperiod <tau_pressure> tswitch <switch_time> tequil <equilibrium_time>

- :attr:`<temperature>`: The temperature of the AS simulation.
- :attr:`<pmin>` and :attr:`<pmax>`: The pressure range of the AS simulation.
- :attr:`<pressure_control>`: Please refer to MTTK ensemble for more information.
- :attr:`<tau_temperature>` and :attr:`<tau_pressure>`: These parameters are optional. Please refer to MTTK ensemble for more information.
- :attr:`<equilibrium_time>`: The number timesteps to equilibrate the system.
- :attr:`<switch_time>`: The number timesteps to vary pressure.

If you do not specify :attr:`<equilibrium_time>` and :attr:`<switch_time>`, they will be automatically set in a 1:4 ratio.

Example
-------

.. code-block:: rst

    ensemble ti_as temp 300 aniso 10 30 tswitch 10000

This command switch pressure from 10 to 30 GPa in 10000 timesteps.

Output file
-----------

This command will produce a csv file. The columns are pressure and volume per atom.

The following code can help you calculate the Gibbs free energy:

.. code-block:: python

    from pandas import read_csv
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.units import kB, GPa
    import yaml
    from scipy.integrate import cumtrapz

    with open("ti_spring_300.yaml", "r") as f:
        y =  yaml.safe_load(f)

    T0 = y["T"]
    G0 = y["G"]

    ti = read_csv("ti_as.csv")
    n = int(len(ti)/2)
    forward = ti[:n]
    backward = ti[n:][::-1]
    backward.reset_index(inplace=True)
    p = forward["p"]
    V1 = forward["V"]
    V2 = backward["V"]

    w = (cumtrapz(V1,p,initial=0) + cumtrapz(V2,p,initial=0))*0.5

    G = G0 + w
    plt.plot(p/GPa, G, label="AS")
    plt.legend()
    plt.xlabel("P (GPa)")
    plt.ylabel("G (eV/atom)")