.. _kw_ensemble_ti_nep:

:attr:`ensemble` (TI_NEP)
=========================

This keyword is used to set up a nonequilibrium thermodynamic integration integrator between two :term:`NEP` models. Please check [Freitas2016]_ for more details on the nonequilibrium switching scheme.

The system is driven by the Hamiltonian

.. math::

   H(\lambda) = (1-\lambda) U_1 + \lambda U_2,

where :math:`U_1` and :math:`U_2` are the potential energies from the first (NEP1) and the second (NEP2) :ref:`potential <kw_potential>` specified in the :attr:`run.in` file.
Lambda is switched from 0 to 1 (forward) and back from 1 to 0 (backward), and the Helmholtz free energy difference :math:`F_{\rm NEP1} - F_{\rm NEP2}` is obtained from the average of the forward and backward work.
This is an NVT ensemble with a Langevin thermostat, so the free energy difference is evaluated at the volume of the simulation box.

Syntax
------

The parameters can be specified as follows::

    ensemble ti_nep temp <temperature> tperiod <tau_temperature> tequil <equilibrium_time> tswitch <switch_time>

- :attr:`<temperature>`: The temperature of the simulation.
- :attr:`<tau_temperature>`: This parameter is optional, and defaults to ``100``. It determines the period of the thermostat in units of the timestep. It determines how strongly the system is coupled to the thermostat.
- :attr:`<equilibrium_time>`: The number timesteps to equilibrate the system at :math:`\lambda=0` and at :math:`\lambda=1`.
- :attr:`<switch_time>`: The number timesteps to vary lambda from 0 to 1 (and from 1 to 0).

If you do not specify :attr:`<equilibrium_time>` and :attr:`<switch_time>`, they will be automatically set in a 1:4 ratio.
If you specify them, you must specify both, and the number of steps in the :ref:`run keyword <kw_run>` should be 2 × (:attr:`<equilibrium_time>` + :attr:`<switch_time>`).

Exactly two NEP potentials are needed in the :attr:`run.in` file.
The first one is NEP1 (:math:`\lambda=0`) and the second one is NEP2 (:math:`\lambda=1`).
Only the forces are mixed during the switching; the potential energies of NEP1 and NEP2 are kept separately, because both are needed to compute the free energy difference.
The potential energy and pressure written by :ref:`dump_thermo <kw_dump_thermo>` are therefore those of NEP1 during the whole run.

Example
-------

.. code-block:: rst

    potential nep1.txt
    potential nep2.txt
    velocity 300
    ensemble ti_nep temp 300 tequil 10000 tswitch 40000
    run 100000

This command equilibrates for 10000 timesteps with NEP1, switches lambda from 0 to 1 in 40000 timesteps, equilibrates for 10000 timesteps with NEP2, and switches lambda back from 1 to 0 in 40000 timesteps.

Example: switching between defect states
----------------------------------------

This ensemble has been used to compute the free energy difference between two charge states of a point defect [Hainer2025]_ and between polaronic states [Berger2026]_.
One NEP model is trained for one charge state and then relabeled for the other charge state; the original and the relabeled model are given as NEP1 and NEP2.
The resulting :attr:`F_diff` is the raw Helmholtz free energy difference between the two charge states, which goes into the expression for the formation free energy of the defect.

Output file
-----------

This command will produce a csv file :attr:`ti_nep.csv`. The columns are lambda, dlambda, potential energy of NEP1 and potential energy of NEP2 (eV/atom). The file is only written during the switching.

This command will also produce a yaml file :attr:`ti_nep.yaml`, which contains the Helmholtz free energy difference :attr:`F_diff` (:math:`F_{\rm NEP1} - F_{\rm NEP2}`, eV/atom) and the temperature.

Both files are overwritten by a subsequent :attr:`ti_nep` run in the same directory, so use separate directories for separate calculations.
