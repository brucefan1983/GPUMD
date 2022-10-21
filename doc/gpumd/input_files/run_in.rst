.. _run_in:
.. index::
   single: gpumd input files; Simulation protocol
   single: run.in

Simulation protocol (``run.in``)
--------------------------------

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
* Then, if needed, use the :ref:`minimize <kw_minimize>` keyword to minimize the energy of the whole system.
* Then one can use the following keywords to do carry out static calculations:

  * Use the :ref:`compute_cohesive <kw_compute_cohesive>` keyword to compute the cohesive energy curve.
  * Use the :ref:`compute_elastic <kw_compute_elastic>` keyword to compute the elastic constants.
  * Use the :ref:`compute_phonon <kw_compute_phonon>` keyword to compute the phonon dispersions.
* Then, if one wants to carry out :term:`MD` simulations, one has to set up the initial velocities using the :ref:`velocity <kw_velocity>` keyword and carry out a number of :term:`MD` runs as follows:

  * Specify an integrator using the :ref:`ensemble <kw_ensemble>` keyword and optionally add keywords to further control the evolution and measurement processes. 
  * Use the :ref:`run <kw_run>` keyword to run a number of :term:`MD` steps according to the above settings. 
  * The last two steps can be repeated.

The following tables provide an overview of the different keywords.
A complete list can also be found :ref:`here <input_parameters>`.
The last two columns indicate whether the command is executed immediately (*Exec.*) and whether it is propagted from one ``run`` commandto the next (*Prop.*).

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
   * - :ref:`velocity <kw_velocity>`
     - Set the initial velocities
     - Yes
     - N/A
   * - :ref:`potential <kw_potential>`
     - Set up the interaction model
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
   * - :ref:`compute_gkma <kw_compute_gkma>`
     - Compute the modal heat current using the :term:`GKMA` method
     - No
     - No
   * - :ref:`compute_hac <kw_compute_hac>`
     - Compute the thermal conductivity using the :term:`EMD` method
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
   * - :ref:`compute_phonon <kw_compute_phonon>`
     - Compute the phonon dispersion
     - Yes
     - N/A
   * - :ref:`compute_sdc <kw_compute_sdc>`
     - Compute the self-diffusion coefficient (:term:`SDC`)
     - No
     - No
   * - :ref:`compute_shc <kw_compute_shc>`
     - Compute the spectral heat current (:term:`SHC`)
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
   * - :ref:`dump_exyz <kw_dump_exyz>`
     - Write positions and other quantities in `extended XYZ format <https://github.com/libAtoms/extxyz>`_
     - No
     - No
   * - :ref:`dump_force <kw_dump_force>`
     - Write the atomic forces
     - No
     - No
   * - :ref:`dump_position <kw_dump_position>`
     - Write the atomic positions
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
   * - :ref:`dump_thermo <kw_dump_thermo>`
     - Write thermodynamic quantities
     - No
     - No
   * - :ref:`dump_velocity <kw_dump_velocity>`
     - Write the atomic velocities
     - No
     - No
