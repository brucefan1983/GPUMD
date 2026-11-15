.. _gpumd_executable:
.. index::
   single: gpumd executable

gpumd executable
****************

The ``gpumd`` executable runs molecular dynamics simulations and performs static calculations using an interatomic potential.

For a molecular dynamics simulation:

1. Prepare :ref:`model.xyz <model_xyz>` with the simulation cell, atomic positions, and any required per-atom properties.
2. Choose a :ref:`potential model <potentials>` appropriate for the system and load it with :ref:`potential <kw_potential>` in ``run.in``.
3. Write the simulation protocol in :ref:`run.in <run_in>`, including the ensemble, time step, measurements, and output commands.
   The :ref:`multiple-run example <multiple_runs>` shows separate equilibration and production stages.
4. Run ``gpumd`` in the directory containing the input files, then inspect :ref:`thermo.out <thermo_out>` and the other :doc:`output files <output_files/index>` selected in ``run.in``.

For static calculations, see :ref:`energy minimization <kw_minimize>`, :ref:`elastic constants <kw_compute_elastic>`, and :ref:`phonon dispersions <kw_compute_phonon>`.
To continue a simulation from a saved configuration, see :ref:`restart.xyz <restart_xyz>`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   input_files/index
   input_parameters/index
   output_files/index
