.. _kw_ensemble_ti_as:

:attr:`ensemble` (TI_AS)
========================

This keyword is used to set up a nonequilibrium thermodynamic integration integrator with Adiabatic switching (AS) path. It calculates Gibbs free energy along an isothermal path. See [Cajahuaringa2022]_ for more details.

Syntax
------

::

    ensemble ti_as temp <temperature> tperiod <tau_temperature> <pressure_control> <pmin> <pmax> pperiod <tau_pressure> tswitch <switch_time> tequil <equilibrium_time>

Parameters
----------

- :attr:`<temperature>`: The temperature of the AS simulation, in kelvin.
- :attr:`<pmin>` and :attr:`<pmax>`: The initial and final pressures of the forward path, in GPa. Either increasing or decreasing pressure can be used.
- :attr:`<pressure_control>`: See :ref:`MTTK parameters <kw_ensemble_mttk>`.
- :attr:`<tau_temperature>` and :attr:`<tau_pressure>`: These parameters are optional. See :ref:`MTTK parameters <kw_ensemble_mttk>`.
- :attr:`<equilibrium_time>`: The number of MD steps in each equilibration stage; a non-negative integer.
- :attr:`<switch_time>`: The number of MD steps in each switching stage, including both endpoints; an integer of at least 2.

Specify both ``tequil`` and ``tswitch``, or omit both.
When omitted, they are set to ``floor(0.1 * N)`` and ``floor(0.4 * N)``, respectively, where ``N`` is the number of run steps; the resulting ``tswitch`` must still be at least 2.
The path consists of initial equilibration, forward switching, equilibration at the other endpoint, and backward switching.
The run must contain at least ``2 * (tequil + tswitch)`` steps.
Any remaining steps keep the final target and do not add switching data.

Examples
--------

.. code-block:: text

    ensemble ti_as temp 300 aniso 10 30 tswitch 10000 tequil 1000
    run 22000

This run equilibrates for 1000 steps at 10 GPa, switches to 30 GPa in 10000 steps, equilibrates there for 1000 steps, and switches back to 10 GPa in 10000 steps, all at 300 K.

Output file
-----------

``ti_as.csv`` contains ``2 * tswitch`` rows after the header: the forward path followed by the backward path.
The columns ``p`` and ``V`` are the target pressure in eV/Å³ and the volume per atom in Å³/atom.
Equilibration steps are not written. Each subsequent TI_AS run overwrites this file.

The following example uses ``ti_spring.yaml`` from a reference calculation at the same initial temperature and pressure.
Copy that file into the working directory before running the script.
The reference Gibbs free energy and the resulting curve are in eV/atom.

.. code-block:: python

    from pandas import read_csv
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.units import GPa
    import yaml
    from scipy.integrate import cumulative_trapezoid

    with open("ti_spring.yaml", "r") as f:
        y =  yaml.safe_load(f)

    G0 = y["G"]

    ti = read_csv("ti_as.csv")
    n = 10000  # tswitch in the example above
    if len(ti) != 2*n:
        raise ValueError("Expected two complete switching paths.")
    forward = ti.iloc[:n]
    backward = ti.iloc[n:].iloc[::-1]
    if not np.allclose(forward["p"], backward["p"], rtol=2e-6, atol=1e-10):
        raise ValueError("Forward and backward grids do not match.")
    p = forward["p"].to_numpy()
    V1 = forward["V"].to_numpy()
    V2 = backward["V"].to_numpy()

    w = (cumulative_trapezoid(V1,p,initial=0) + cumulative_trapezoid(V2,p,initial=0))*0.5

    G = G0 + w
    plt.plot(p/GPa, G, label="AS")
    plt.legend()
    plt.xlabel("P (GPa)")
    plt.ylabel("G (eV/atom)")
