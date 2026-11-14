.. _run_in:
.. index::
   single: gpumd input files; Simulation protocol
   single: run.in

Simulation protocol (``run.in``)
********************************

The ``run.in`` file defines the simulation protocol.
For ordinary MD, GPUMD executes its commands in file order.

File format
===========

Write one command per line::

   keyword parameter_1 parameter_2 ...

Blank lines are ignored, and ``#`` starts a comment that extends to the end of the line.
An invalid command causes GPUMD to report an error and stop.

Command order
=============

For an ordinary MD simulation:

1. If needed, use :ref:`replicate <kw_replicate>` before loading a potential.
2. Load the interaction model with :ref:`potential <kw_potential>`.
3. If needed, relax the structure with :ref:`minimize <kw_minimize>`.
4. Set the initial temperature with :ref:`velocity <kw_velocity>` if ``model.xyz`` does not supply velocities.
   The default is 300 K. If velocities are supplied, GPUMD uses them and ignores this initial-temperature setting.
5. Choose the :ref:`ensemble <kw_ensemble>` and :ref:`time step <kw_time_step>`, then add the required control, measurement, and output commands.
6. Execute :ref:`run <kw_run>` for the desired number of steps.

For each subsequent run, choose the ensemble and repeat the commands needed for that run.
The atomic positions, velocities, and cell continue from the preceding run; see the :ref:`multiple-run example <multiple_runs>`.
To continue from a saved configuration in a new simulation, see :ref:`restart.xyz <restart_xyz>`.

Static calculations such as :ref:`cohesive-energy curves <kw_compute_cohesive>`, :ref:`elastic constants <kw_compute_elastic>`, and :ref:`phonon dispersions <kw_compute_phonon>` execute when their commands are read, after the potential has been loaded.
For calculations with several NEP models, see :ref:`dump_observer <kw_dump_observer>`.

Replica dynamics use :ref:`multi_replica <kw_multi_replica>`, a :ref:`restricted command set <replica_input_parameters>`, and a single final ``run`` command.
See the :ref:`replica input example <replica_run_in>`.

Keyword overview
================

The tables below summarize when commands take effect and whether their settings carry over to the next run.
The :ref:`keyword reference <gpumd_input_parameters>` gives their syntax and parameters.

* *Exec.*: ``Yes`` means that the command executes immediately when read.
* *Prop.*: ``Yes`` means that the setting carries over to the next ``run``.
  Repeat a non-propagating setting before each run that needs it.
* *N/A*: there is no per-run setting to carry over.
  The effect of an immediately executed command, such as a loaded potential or changed cell, remains in place.

For :ref:`time_step <kw_time_step>`, the current step size carries over, but the optional displacement limit must be specified again.

Simulation setup
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: auto

   * - Keyword
     - Brief description
     - Exec.
     - Prop.
   * - :ref:`replicate <kw_replicate>`
     - Replicate the simulation model
     - Yes
     - N/A
   * - :ref:`velocity <kw_velocity>`
     - Set the initial velocities
     - Yes
     - N/A
   * - :ref:`correct_velocity <kw_correct_velocity>`
     - Remove linear and angular momenta at regular intervals
     - No
     - No
   * - :ref:`potential <kw_potential>`
     - Set up the interaction model
     - Yes
     - N/A
   * - :ref:`dftd3 <kw_dftd3>`
     - Add the DFT-D3 dispersion correction to the NEP model
     - Yes
     - N/A
   * - :ref:`kspace <kw_kspace>`
     - Specify the reciprocal-space method used by the interaction model
     - Yes
     - N/A
   * - :ref:`compute_extrapolation <kw_compute_extrapolation>`
     - Monitor the NEP extrapolation grade during an :term:`MD` run
     - No
     - No
   * - :ref:`change_box <kw_change_box>`
     - Change the box
     - Yes
     - N/A
   * - :ref:`deform <kw_deform>`
     - Deform the simulation box
     - No
     - No
   * - :ref:`time_step <kw_time_step>`
     - Specify the integration time step
     - No
     - Yes
   * - :ref:`ensemble <kw_ensemble>`
     - Specify the integrator for a :term:`MD` run
     - No
     - No
   * - :ref:`add_force <kw_add_force>`
     - Add external forces to selected atoms
     - No
     - No
   * - :ref:`add_efield <kw_add_efield>`
     - Apply an electric field to selected atoms
     - No
     - No
   * - :ref:`add_mfield <kw_add_mfield>`
     - Apply a magnetic field to an atom group or fractional region
     - No
     - No
   * - :ref:`add_spring <kw_add_spring>`
     - Add spring interactions to selected atom groups
     - No
     - No
   * - :ref:`deposit <kw_deposit>`
     - Add atoms periodically during a deposition run
     - No
     - No
   * - :ref:`multi_replica <kw_multi_replica>`
     - Select REMD or PRD for the input file
     - No
     - N/A
   * - :ref:`electron_stop <kw_electron_stop>`
     - Apply electronic stopping forces to high-energy atoms
     - No
     - No
   * - :ref:`fix <kw_fix>`
     - Fix (freeze) atoms
     - No
     - No
   * - :ref:`move <kw_move>`
     - Move selected atoms with a constant velocity
     - No
     - No
   * - :ref:`mc <kw_mc>`
     - Carry out Monte Carlo trial steps during an :term:`MD` run
     - No
     - No
   * - :ref:`plumed <kw_plumed>`
     - Invoke the PLUMED plugin during an :term:`MD` run
     - No
     - No

Actions
^^^^^^^

.. list-table::
   :header-rows: 1

   * - Keyword
     - Brief description
     - Exec.
     - Prop.
   * - :ref:`minimize <kw_minimize>`
     - Perform an energy minimization
     - Yes
     - N/A
   * - :ref:`minimize spin_fire <spin_fire_minimize>`
     - Minimize magnetic moments, optionally together with atomic positions and cell strain
     - Yes
     - N/A
   * - :ref:`run <kw_run>`
     - Run a number of :term:`MD` steps
     - Yes
     - No
   * - :ref:`compute <kw_compute>`
     - Compute some time and space-averaged quantities
     - No
     - No
   * - :ref:`compute_chunk <kw_compute_chunk>`
     - Compute time-averaged quantities in dynamic spatial bins
     - No
     - No
   * - :ref:`compute_adf <kw_compute_adf>`
     - Compute the angular distribution function (:term:`ADF`)
     - No
     - No
   * - :ref:`compute_angular_rdf <kw_compute_angular_rdf>`
     - Compute the angular-dependent radial distribution function (:term:`ARDF`)
     - No
     - No
   * - :ref:`compute_cohesive <kw_compute_cohesive>`
     - Compute the cohesive energy curve
     - Yes
     - N/A
   * - :ref:`compute_elastic <kw_compute_elastic>`
     - Compute the elastic constants
     - Yes
     - N/A
   * - :ref:`compute_dos <kw_compute_dos>`
     - Compute the phonon density of states (:term:`PDOS`)
     - No
     - No
   * - :ref:`compute_dpdt <kw_compute_dpdt>`
     - Compute the time derivative of the total dipole moment
     - No
     - No
   * - :ref:`compute_gkma <kw_compute_gkma>`
     - Compute the modal heat current using the :term:`GKMA` method
     - No
     - No
   * - :ref:`compute_hac <kw_compute_hac>`
     - Compute the thermal conductivity using the :term:`EMD` method
     - No
     - No
   * - :ref:`compute_ic <kw_compute_ic>`
     - Compute the ionic conductivity (:term:`IC`)
     - No
     - No
   * - :ref:`compute_hnema <kw_compute_hnema>`
     - Compute the modal thermal conductivity using the :term:`HNEMA` method
     - No
     - No
   * - :ref:`compute_hnemd <kw_compute_hnemd>`
     - Compute the thermal conductivity using the :term:`HNEMD` method
     - No
     - No
   * - :ref:`compute_hnemdec <kw_compute_hnemdec>`
     - Compute the multicomponent system thermal conductivity using the :term:`HNEMDEC` method
     - No
     - No
   * - :ref:`compute_orientorder <kw_compute_orientorder>`
     - Compute Steinhardt bond-orientational order parameters
     - No
     - No
   * - :ref:`compute_phonon <kw_compute_phonon>`
     - Compute the phonon dispersion
     - Yes
     - N/A
   * - :ref:`compute_sdc <kw_compute_sdc>`
     - Compute the self-diffusion coefficient (:term:`SDC`)
     - No
     - No
   * - :ref:`compute_msd <kw_compute_msd>`
     - Compute the mean-square displacement (:term:`MSD`)
     - No
     - No
   * - :ref:`compute_rdf <kw_compute_rdf>`
     - Compute the radial distribution function (:term:`RDF`)
     - No
     - No
   * - :ref:`compute_shc <kw_compute_shc>`
     - Compute the spectral heat current (:term:`SHC`)
     - No
     - No
   * - :ref:`compute_viscosity <kw_compute_viscosity>`
     - Compute the stress autocorrelation function and viscosity
     - No
     - No
   * - :ref:`compute_lsqt <kw_compute_lsqt>`
     - Compute electronic transport properties using the :term:`LSQT` method
     - No
     - No

Output
^^^^^^

.. list-table::
   :header-rows: 1

   * - Keyword
     - Brief description
     - Exec.
     - Prop.
   * - :ref:`active <kw_active>`
     - Run on-the-fly active learning, saving structures that exceeds a set threshold maximum force uncertainty over all specified NEP potentials.
     - No
     - No
   * - :ref:`dump_beads <kw_dump_beads>`
     - Write bead-resolved positions and optional velocities and forces for :term:`PIMD`-related runs
     - No
     - No
   * - :ref:`dump_dipole <kw_dump_dipole>`
     - Write dipoles predicted by a separate tensorial NEP model
     - No
     - No
   * - :ref:`dump_observer <kw_dump_observer>`
     - Write positions and other quantities for each of the observing NEP potentials, or the average of them, in the extended XYZ format.
     - No
     - No
   * - :ref:`dump_polarizability <kw_dump_polarizability>`
     - Write polarizabilities predicted by a separate tensorial NEP model
     - No
     - No
   * - :ref:`dump_netcdf <kw_dump_netcdf>`
     - Write the atomic positions in netCDF format
     - No
     - No
   * - :ref:`dump_restart <kw_dump_restart>`
     - Write a restart file
     - No
     - No
   * - :ref:`dump_shock_nemd <kw_dump_shock_nemd>`
     - Write spatial thermodynamic profiles for shock-wave NEMD simulations
     - No
     - No
   * - :ref:`dump_thermo <kw_dump_thermo>`
     - Write thermodynamic quantities
     - No
     - No
   * - :ref:`dump_xyz <kw_dump_xyz>`
     - Write positions and other per-atom quantities in `extended XYZ format <https://github.com/libAtoms/extxyz>`_
     - No
     - No
