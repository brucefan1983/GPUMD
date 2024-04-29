.. _kw_ensemble_ti_rs:

:attr:`ensemble` (TI_RS)
============================

This keyword is used to set up a nonequilibrium thermodynamic integration integrator with Reversible Scaling (RS) path. It calculates Gibbs free energy along an isobaric path. Please check [Koning2001]_ and [Freitas2016]_ for more details.

Syntax
------

The parameters can be specified as follows::

    ensemble ti_rs temp <tmin> <tmax> tperiod <tau_temperature> <pressure_control> <pressure> pperiod <tau_pressure> tswitch <switch_time> tequil <equilibrium_time>

- :attr:`<tmin>` and :attr:`<tmax>`: The temperature range of the RS simulation.
- :attr:`<pressure_control>` and :attr:`<pressure>`: The pressure of RS simualtion. Please refer to MTTK ensemble for more information.
- :attr:`<tau_temperature>` and :attr:`<tau_pressure>`: These parameters are optional. Please refer to MTTK ensemble for more information.
- :attr:`<equilibrium_time>`: The number timesteps to equilibrate the system.
- :attr:`<switch_time>`: The number timesteps to vary lambda.

If you do not specify :attr:`<equilibrium_time>` and :attr:`<switch_time>`, they will be automatically set in a 1:4 ratio.

Example
-------

.. code-block:: rst

    ensemble ti_rs temp 300 3000 aniso 10 tswitch 10000 tequil 1000

This command switch lambda for 10000 timesteps.

Output file
-----------

This command will produce a csv file. The columns are lambda, dlambda, and enthalpy (eV/atom).

The following code can help you calculate the Gibbs free energy:

.. code-block:: python

    import yaml
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import read_csv
    from scipy.integrate import cumtrapz
    from ase.units import kB

    # you need to run a ti_spring simulation at t_min
    with open("ti_spring_300.yaml", "r") as f:
        y =  yaml.safe_load(f)

    T0 = y["T"]
    G0 = y["G"]

    rs = read_csv("ti_rs.csv")
    n = int(len(rs)/2)
    forward = rs[:n]
    backward = rs[n:][::-1]
    backward.reset_index(inplace=True)
    dl = forward["dlambda"]
    l = forward["lambda"]
    H1 = forward["enthalpy"]
    H2 = backward["enthalpy"]
    T = T0/l

    w = (cumtrapz(H1,l,initial=0) + cumtrapz(H2,l,initial=0))*0.5

    G = (G0 + 1.5*kB*T0*np.log(l) + w)/l 
    plt.plot(T, G, label="RS")
    plt.legend()
    plt.xlabel("T (K)")
    plt.ylabel("G (eV/atom)")