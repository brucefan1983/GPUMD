.. _restart_xyz:
.. index::
   single: restart.xyz (output file)

``restart.xyz``
===============

This file stores the simulation cell and atomic data, including positions and velocities, for continuing a simulation.
It is generated when invoking the :ref:`dump_restart keyword <kw_dump_restart>`.

File format
-----------
This file has the same format as the :ref:`simulation model input file <model_xyz>`.

* The output mode for this file is overwrite.

Continuing a simulation
-----------------------

Copy ``restart.xyz`` to ``model.xyz`` and provide the potential and a ``run.in`` for the next stage.
GPUMD uses the saved velocities, so the initial temperature in the :ref:`velocity command <kw_velocity>` is ignored and this command can be omitted.
The :ref:`ensemble <kw_ensemble>` target temperature still applies.
Set the desired ensemble, time step and output commands in the new ``run.in``.
This provides a continuation from the saved configuration; an identical trajectory to an uninterrupted run is not guaranteed.
