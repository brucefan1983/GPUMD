.. _kw_ensemble_ti_rs:

:attr:`ensemble` (TI_RS)
========================

This keyword is used to set up a nonequilibrium thermodynamic integration integrator with Reversible Scaling (RS) path. It calculates Gibbs free energy along an isobaric path. See [Koning2001]_ and [Freitas2016]_ for more details.

Syntax
------

::

    ensemble ti_rs temp <tmin> <tmax> tperiod <tau_temperature> <pressure_control> <pressure> pperiod <tau_pressure> tswitch <switch_time> tequil <equilibrium_time>

Parameters
----------

- :attr:`<tmin>` and :attr:`<tmax>`: The initial and final physical temperatures of the forward path, in kelvin.
- :attr:`<pressure_control>` and :attr:`<pressure>`: The pressure-control mode and the physical pressure in GPa. See :ref:`MTTK parameters <kw_ensemble_mttk>`.
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

    ensemble ti_rs temp 300 3000 aniso 10 tswitch 10000 tequil 1000
    run 22000

This run equilibrates for 1000 steps at :math:`\lambda=1`, switches to :math:`\lambda=300/3000=0.1` in 10000 steps, equilibrates there for 1000 steps, and switches back to 1 in 10000 steps.
The thermostat temperature remains 300 K; the corresponding physical temperature is :math:`T=300/\lambda`.

Output file
-----------

``ti_rs.csv`` contains ``2 * tswitch`` rows after the header: the forward path followed by the backward path.
The columns are ``lambda``, ``dlambda`` (the derivative of lambda with respect to the MD step index), and ``enthalpy`` (the unscaled potential energy plus the physical pressure times volume, per atom, in eV/atom).
Equilibration steps are not written. Each subsequent TI_RS run overwrites this file.

The following example uses ``ti_spring.yaml`` from a reference calculation at the same initial temperature and pressure.
Copy that file into the working directory before running the script.
The reference Gibbs free energy and the resulting curve are in eV/atom.

.. code-block:: python

    import yaml
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import read_csv
    from scipy.integrate import cumulative_trapezoid
    from ase.units import kB

    with open("ti_spring.yaml", "r") as f:
        y =  yaml.safe_load(f)

    T0 = y["T"]
    G0 = y["G"]

    rs = read_csv("ti_rs.csv")
    n = 10000  # tswitch in the example above
    if len(rs) != 2*n:
        raise ValueError("Expected two complete switching paths.")
    forward = rs.iloc[:n]
    backward = rs.iloc[n:].iloc[::-1]
    if not np.allclose(forward["lambda"], backward["lambda"], rtol=2e-6, atol=1e-10):
        raise ValueError("Forward and backward grids do not match.")
    l = forward["lambda"].to_numpy()
    H1 = forward["enthalpy"].to_numpy()
    H2 = backward["enthalpy"].to_numpy()
    T = T0/l

    w = (cumulative_trapezoid(H1,l,initial=0) + cumulative_trapezoid(H2,l,initial=0))*0.5

    G = (G0 + 1.5*kB*T0*np.log(l) + w)/l
    plt.plot(T, G, label="RS")
    plt.legend()
    plt.xlabel("T (K)")
    plt.ylabel("G (eV/atom)")
