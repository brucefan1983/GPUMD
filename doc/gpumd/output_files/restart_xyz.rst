.. _restart_xyz:
.. index::
   single: restart.xyz (output file)

``restart.xyz``
===============

The :ref:`dump_restart command <kw_dump_restart>` writes the simulation cell and atomic data, including positions and velocities, to this file at the specified interval.
Each write overwrites the previous contents.

File format
-----------
This file has the same format as the :ref:`simulation model input file <model_xyz>`.

Continuing a simulation
-----------------------

Copy ``restart.xyz`` to ``model.xyz`` and provide the potential and a ``run.in`` for the next stage.
GPUMD uses the saved velocities, so the initial temperature in the :ref:`velocity command <kw_velocity>` is ignored and this command can be omitted.
The :ref:`ensemble <kw_ensemble>` target temperature still applies.
Set the desired ensemble, time step and output commands in the new ``run.in``.
This provides a continuation from the saved configuration; an identical trajectory to an uninterrupted run is not guaranteed.
