.. _replica_remd_tp:
.. index::
   single: multi_replica remd_tp (keyword in run.in)

:attr:`multi_replica remd_tp`
==============================

Run ``gpumd`` with :ref:`multi_replica <kw_multi_replica>` in ``run.in``.
This mode selects two-dimensional temperature-pressure replica exchange in the isothermal-isobaric ensemble [Okabe2001]_.
The :math:`M = M_T \times M_P` replicas sit on a temperature-pressure grid.
Each replica runs NPT dynamics with the Bussi-Donadio-Parrinello thermostat and the stochastic cell rescaling barostat [Bernetti2020]_.

Syntax
------

::

  multi_replica remd_tp replicas <number> [<per_gpu>] exchange <interval>
                temp <T_min> <T_max> <M_T> | temp <filename>
                pressure <P_min> <P_max> <M_P> | pressure <filename> | pressure_grid <filename>
                [spacing <geometric|linear>] [equilibrate <steps>] [resume <prefix>]

In this replica mode, the ``ensemble`` command uses the following syntax
(the two pressures specify grid endpoints rather than a pressure ramp)::

  ensemble npt_scr <T_min> <T_max> <tau_T> <P_min> <P_max> <bulk_modulus> <tau_p>
                   [couple <dirs>] [pcouple_interval <steps>]

* ``replicas`` must equal :math:`M_T \times M_P`.
* ``temp``: temperatures in K. ``spacing`` (default ``geometric``) applies to the generated temperatures.
  :math:`M_T = 1` (with :math:`T_{min} = T_{max}`) gives pressure-only exchange.
* ``pressure``: pressures in GPa, linearly spaced. Negative pressures are allowed.
  :math:`M_P = 1` gives temperature-only exchange at constant pressure.
* A file contains one value per entry, strictly increasing; ``#`` starts a comment.
* ``pressure_grid``: a sheared grid. The file has one line per temperature (in increasing temperature order) with :math:`M_P` strictly increasing pressures (GPa).
  Each temperature row may use different pressures, for example centred on an estimated Widom or coexistence line, so that the replicas concentrate where the response functions are large.
  Temperature-direction exchanges then change both :math:`T` and :math:`P`; the acceptance rule below is exact for any pair of conditions.
  The ``npt_scr`` pressures must be the lowest and highest pressures in the file.
* ``tau_T`` and ``tau_p`` are in MD steps; ``bulk_modulus`` (GPa) sets the barostat response and does not affect the sampled distribution.
* ``couple`` lists the periodic directions scaled together by the barostat (default ``xyz``, isotropic). ``couple z`` controls the axial pressure only, ``couple xy`` the lateral pressure only.
* ``pcouple_interval``: MD steps between barostat updates (default 10). ``tau_p`` must be at least twice this value.

Replica labels are ``label = p_index * M_T + t_index``.
Exchange rounds cycle through even temperature pairs, even pressure pairs, odd temperature pairs, and odd pressure pairs.
A swap of the conditions of replicas :math:`i` and :math:`j`, at conditions :math:`a` and :math:`b`, is accepted with probability

.. math::

   \min\left\{1, \exp\left[(\beta_a - \beta_b)(E_i - E_j) + (\beta_a P_a - \beta_b P_b)(V_i - V_j)\right]\right\},

where :math:`E` is the potential energy and :math:`V` the volume.
Temperature swaps rescale velocities by :math:`\sqrt{T_{new}/T_{old}}`; coordinates and boxes stay with their replica.

Barostat
--------

With :math:`\epsilon = \ln V`, each update integrates

.. math::

   d\epsilon = -\frac{\beta_T}{\tau_p}\left(P_0 - P\right)dt + \sqrt{\frac{2k_BT\beta_T}{V\tau_p}}\,dW

over ``pcouple_interval`` steps, with :math:`\beta_T` the inverse of ``bulk_modulus``.
The coupled box lengths and positions are scaled by :math:`e^{d\epsilon/d}` (:math:`d` coupled directions) and velocities by its inverse.
The Ito correction cancels the :math:`k_BT/V` term when changing variables from volume to its logarithm (Eq. 5 of [Bernetti2020]_).
The pressure is measured after BDP velocity rescaling at the end of an update step and the box is rescaled before the next step, so the energies and volumes used for exchanges belong to the same configuration.
If a temperature exchange occurs at that boundary, the pressure is refreshed using the exchanged velocities and the pending box update is retained.
This is the Euler SCR integration scheme, with finite-step sampling error; forces are evaluated once per MD step.
Initial velocities are sampled from the canonical distribution with zero total linear momentum; the thermostat uses :math:`3N-3` degrees of freedom.

Example
-------

256 water molecules on a 6 x 4 grid::

  potential nep.txt
  time_step 0.5
  ensemble npt_scr 250 300 100 0 0.15 2.0 1000
  multi_replica remd_tp replicas 24 exchange 200 temp 250 300 6 pressure 0 0.15 4 equilibrate 20000
  dump_xyz -1 0 2000 trajectory.xyz
  run 200000

Output files
------------

* ``remd_tp_conditions.out``: label, grid indices, temperature (K), pressure (GPa).
* ``remd_tp_exchange.out``: one row per attempted pair: step, direction (``T``/``P``), replicas, labels, :math:`T`, :math:`P`, :math:`E`, :math:`V`, log acceptance ratio, probability, accepted.
* ``remd_tp_status.out``: step and the condition label of each replica.
* ``remd_tp_thermo.out``: step, then condition label, potential energy (eV), and volume (Å\ :sup:`3`) of each replica; written at each advancement boundary, including every exchange.
  Collect statistics by condition label, not by replica index.
* ``remd_tp_pairs.out``: attempts and acceptance for every neighbouring pair.
* Trajectories (``dump_xyz``) and restart files (``remd_tp_restart.meta``, ``remd_tp_replica_<index>_restart.xyz``) include the current box; continue with ``resume remd_tp``.

Caveats
-------

* Energy NEP models only (no qNEP); orthogonal boxes and at least two atoms are required.
* Each replica has its own box, so batched integration is disabled in this mode.
* Continuation restores atomic states, condition assignments, and random-number states.
  It does not promise the trajectory or exchange outcomes of an uninterrupted run.
* REMD-TP restart format 3 is required. Older checkpoints use the previous SCR equation and thermostat convention and are rejected; do not change their version number to bypass this check.
* On restart, PBC, atom properties, conditions and all metadata must match. The restored box must be orthogonal with positive finite lengths; uncoupled lengths must match ``model.xyz``, while coupled lengths are restored from the checkpoint.
* Pressure spacing must match volume fluctuations: neighbouring pressures should satisfy :math:`\Delta P\,\sigma_V \lesssim k_BT`.
* Too small ``bulk_modulus`` or ``tau_p`` makes the barostat unstable; the run stops if a box length changes by more than 5% in one update.
