.. _kw_run:
.. index::
   single: run (keyword in run.in)

:attr:`run`
===========

Run a number of :term:`MD` steps according to the settings specified for the current run.

With :ref:`multi_replica <kw_multi_replica>`, specify one ``run`` command at the end of ``run.in``.
See :ref:`replica_kw_run` for how steps are counted in REMD, PRD, and resumed runs.

Syntax
------

::

  run <number_of_steps>

Parameters
----------

``number_of_steps`` is a positive integer giving the number of MD steps.

Output
------

Output is controlled by the measurement and dump commands enabled for this run; see :doc:`../output_files/index`.

Examples
--------

To run one million steps, just write::

  run 1000000

Notes
-----

.. _multiple_runs:

Multiple runs
^^^^^^^^^^^^^

For ordinary MD, successive runs in the same ``run.in`` continue from the current positions, velocities and simulation cell.
The potential model remains loaded, and the current :ref:`time step <kw_time_step>` is retained unless changed.
Specify exactly one :ref:`ensemble <kw_ensemble>` before each run.
Repeat the required measurement, output and control commands, such as ``dump_thermo``, ``compute_rdf``, ``fix`` and ``deform``, before each run that uses them.

Sampling and output intervals are counted from the beginning of each run.
Run-specific averages and correlation functions are accumulated separately for each run, even when their results are appended to the same output file.
Choose a run length that provides enough samples for the requested calculations; the append or overwrite behavior is described for each :doc:`output file </gpumd/output_files/index>`.

For example, the following input uses a fixed cell, with an equilibration stage at 300 K followed by NVE production::

  potential nep.txt
  velocity 300
  time_step 1

  # Equilibration
  ensemble nvt_bdp 300 300 100
  dump_thermo 100
  run 10000

  # Production
  ensemble nve
  dump_thermo 100
  dump_xyz 1000 trajectory.xyz
  dump_restart 10000
  run 100000

Use a potential appropriate for the supplied ``model.xyz`` and choose the time step and run lengths for the system being studied.
The second run uses the velocities reached at the end of equilibration, so ``velocity`` is not repeated.
``dump_thermo`` is repeated because output settings apply to one run; ``dump_xyz`` and ``dump_restart`` are enabled only for production.
The generated velocities have an initial temperature of 300 K only when ``model.xyz`` contains no velocity data.
