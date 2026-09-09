.. _run_in:
.. index::
   single: gpumd input files; Simulation protocol
   single: run.in

Simulation protocol (``run.in``)
********************************

The ``run.in`` file is used to define the simulation protocol.
The code will execute the commands in this file one by one.
If the code encounters an invalid command in this file on start-up, it will report an error message and terminate.
In this input file, blank lines and lines starting with ``#`` are ignored.
One can thus write comments after ``#``.
All other lines should be of the form:

.. code::

   keyword parameter_1 parameter_2 ...

The overall structure of a ``run.in`` file is as follows:

* First, set up the potential model using the :ref:`potential <kw_potential>` keyword.
  * Multiple NEP potentials may be used using the :ref:`dump_observer <kw_dump_observer>` keyword. These NEP potentials may either be used to evaluate energy, forces and virials along the trajectory, or averaged together to run the MD on an average PES. 
* Then, if needed, use the :ref:`minimize <kw_minimize>` keyword to minimize the energy of the whole system.
* Then one can use the following keywords to carry out static calculations:

  * Use the :ref:`compute_cohesive <kw_compute_cohesive>` keyword to compute the cohesive energy curve.
  * Use the :ref:`compute_elastic <kw_compute_elastic>` keyword to compute the elastic constants.
  * Use the :ref:`compute_phonon <kw_compute_phonon>` keyword to compute the phonon dispersions.
* Then, if one wants to carry out :term:`MD` simulations, one has to set up the initial velocities using the :ref:`velocity <kw_velocity>` keyword and carry out a number of :term:`MD` runs as follows:

  * Specify an integrator using the :ref:`ensemble <kw_ensemble>` keyword and optionally add keywords to further control the evolution and measurement processes. 
  * Use the :ref:`run <kw_run>` keyword to run a number of :term:`MD` steps according to the above settings. 
  * The last two steps can be repeated.

The following tables provide an overview of the different keywords.
A complete list can also be found :ref:`here <gpumd_input_parameters>`.
The last two columns indicate whether the command is executed immediately (*Exec.*) and whether it is propagated from one ``run`` command to the next (*Prop.*).

Simulation setup
================

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: auto

   * - Keyword
     - Brief description
     - Exec.
     - Prop.
   * - :ref:`velocity <kw_velocity>`
     - Set the initial velocities
     - Yes
     - N/A
   * - :ref:`potential <kw_potential>`
     - Set up the interaction model
     - Yes
     - N/A
   * - :ref:`dftd3 <kw_dftd3>`
     - Add the DFT-D3 dispersion correction to the NEP model
     - Yes
     - N/A
   * - :ref:`change_box <kw_change_box>`
     - Change the box
     - Yes
     - N/A
   * - :ref:`deform <kw_deform>`
     - Deform the simulation box
     - No
     - No
   * - :ref:`ensemble <kw_ensemble>`
     - Specify the integrator for a :term:`MD` run
     - No
     - No
   * - :ref:`fix <kw_fix>`
     - Fix (freeze) atoms
     - No
     - No
   * - :ref:`time_step <kw_time_step>`
     - Specify the integration time step
     - No
     - Yes

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
     - Compute the time derivative of the polarization
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
