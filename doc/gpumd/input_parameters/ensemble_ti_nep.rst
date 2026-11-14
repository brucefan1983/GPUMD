.. _kw_ensemble_ti_nep:

:attr:`ensemble` (TI_NEP)
=========================

This keyword is used to set up a nonequilibrium thermodynamic integration integrator between two :term:`NEP` models. See [Freitas2016]_ for more details on the nonequilibrium switching scheme.

The system is driven by the Hamiltonian

.. math::

   H(\lambda) = (1-\lambda) U_1 + \lambda U_2,

where :math:`U_1` and :math:`U_2` are the potential energies from the first (NEP1) and the second (NEP2) :ref:`potential <kw_potential>` specified in the :attr:`run.in` file.
Lambda is switched from 0 to 1 (forward) and back from 1 to 0 (backward), and the Helmholtz free energy difference :math:`F_{\rm NEP1} - F_{\rm NEP2}` is obtained from the average of the forward and backward work.
This is an NVT ensemble with a Langevin thermostat, so the free energy difference is evaluated at the volume of the simulation box.

Syntax
------

::

    ensemble ti_nep temp <temperature> tperiod <tau_temperature> tequil <equilibrium_time> tswitch <switch_time>

Parameters
----------

- :attr:`<temperature>`: The temperature of the simulation.
- :attr:`<tau_temperature>`: Thermostat coupling period in time steps; optional, with a default of ``100``.
- :attr:`<equilibrium_time>`: The number of time steps to equilibrate the system at :math:`\lambda=0` and at :math:`\lambda=1`.
- :attr:`<switch_time>`: The number of time steps to vary lambda from 0 to 1 (and from 1 to 0).

Specify both ``tequil`` and ``tswitch``, or omit both.
``tequil`` must be a non-negative integer and ``tswitch`` a positive integer.
When omitted, they are set to ``floor(0.1 * N)`` and ``floor(0.4 * N)``, respectively, where ``N`` is the number of run steps; the resulting ``tswitch`` must still be positive.
The run must contain at least ``2 * (tequil + tswitch)`` steps.

Exactly two NEP potentials are needed in the :attr:`run.in` file.
The first one is NEP1 (:math:`\lambda=0`) and the second one is NEP2 (:math:`\lambda=1`).
The potential energy and pressure written by :ref:`dump_thermo <kw_dump_thermo>` are those of NEP1 throughout the run; use ``ti_nep.csv`` for the energies of both models.

Examples
--------

.. code-block:: text

    potential nep1.txt
    potential nep2.txt
    velocity 300
    ensemble ti_nep temp 300 tequil 10000 tswitch 40000
    run 100000

This command equilibrates for 10000 time steps with NEP1, switches lambda from 0 to 1 in 40000 time steps, equilibrates for 10000 time steps with NEP2, and switches lambda back from 1 to 0 in 40000 time steps.

Switching between defect states
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This ensemble has been used to compute the free energy difference between two charge states of a point defect [Hainer2025]_ and between polaronic states [Berger2026]_.
One NEP model is trained for one charge state and then relabeled for the other charge state; the original and the relabeled model are given as NEP1 and NEP2.
The resulting :attr:`F_diff` is the raw Helmholtz free energy difference between the two charge states, which goes into the expression for the formation free energy of the defect.

Output file
-----------

This command will produce a csv file :attr:`ti_nep.csv`. The columns are lambda, dlambda, potential energy of NEP1 and potential energy of NEP2 (eV/atom). The file is only written during the switching.

This command will also produce a yaml file :attr:`ti_nep.yaml`, which contains the Helmholtz free energy difference :attr:`F_diff` (:math:`F_{\rm NEP1} - F_{\rm NEP2}`, eV/atom) and the temperature.

Both files are overwritten by a subsequent :attr:`ti_nep` run in the same directory, so use separate directories for separate calculations.
