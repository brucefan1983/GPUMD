.. _replica_restart:
.. index::
   single: gpumd output files; replica restart files

Replica restart files
=====================

At normal completion, REMD writes ``remd_restart.meta`` and ``remd_replica_<index>_restart.xyz`` for every replica.
NPT REMD uses the same filenames and restores each replica's current box and temperature-pressure assignment.
PRD writes ``prd_restart.meta``, ``prd_replica_<index>_restart.xyz``, and ``prd_basin_restart.xyz``.
Keep the complete set: it stores atomic states, method state, and random-number-generator states.
The replica xyz files store positions in Angstrom, masses in unified atomic mass units, and velocities (``vel``) in Angstrom/fs, with 17 significant digits.
REMD restores its temperature assignments and exchange schedule.
PRD restores its basin reference, running phase, correlation progress, and physical clock.
Random-number-generator states are also restored, but a resumed run is not guaranteed to reproduce an uninterrupted trajectory bit for bit.
Exchange outcomes and event times can differ.

Example
-------

To continue REMD, keep the original input parameters and replace the ``multi_replica`` line with::

  multi_replica remd replicas 8 exchange 1000 temp 300 450 resume remd

For PRD, add ``resume prd`` to its original ``multi_replica`` line.
``run`` specifies additional steps. A prefix can include a directory, such as ``resume ../previous/remd``.
New output uses the usual filenames in the current working directory.

Notes
-----

* Restart files with an incompatible format are rejected.
* ``run.in``, ``model.xyz``, and the potential file are still required.
  Preserve atom count, species, masses, initial model box, potential content, replica count, time step, ensemble, and exchange parameters.
  NPT restores the coupled box lengths from each checkpoint; uncoupled lengths and PBC must match the input model.
  For charge NEP models, also keep the ``kspace`` command: the method and an explicit PPPM spacing are stored and must match.
  A restart that stores ``pppm`` without a spacing matches only a ``kspace`` command without a spacing.
* REMD resume cannot request positive ``equilibrate``.
* Restarts are written at normal completion, not periodically.
  Appended diagnostics and trajectories should correspond to the saved checkpoint.
* PRD checks for events at the end of each block.
  Ending a run can shorten the last block and add an event check, so splitting a run can change event times even without numerical roundoff.

REMD format and extension fields
--------------------------------

The standard metadata header is ``# GPUMD REMD restart metadata``, followed by
``format_version`` and ``mode remd``. Existing NVT inputs continue to read and
write version 5. The NPT extension uses version 6, adds ``ensemble npt_scr`` and
records the temperature-pressure grid, pressure table and SCR coupling parameters.
Common fields retain their existing names. In NPT, ``replica_to_temperature`` and
``temperature_to_replica`` map combined condition labels, rather than just the
temperature-axis index. The two blocks of exchange statistics represent T and P
edges, each indexed by the lower condition label; nonexistent edges stay zero.

Every NPT replica has a ``scr_state`` record containing its integrated MD counter
(including equilibration), pending flag, cached temperature/energy/diagonal
pressures, SCR random-number engine and cached normal variate. BDP and exchange
random-number states are saved by the common REMD writer. These states are
restored before advancement resumes, including a box update pending at the checkpoint.
Cached pressure is in internal units (eV/Angstrom cubed).

Restart XYZ files use ``replica_mode=remd``, ``replica``, ``temperature_label``,
``target_temperature``, ``potential_energy`` and ``restart_step``. The latter is
the completed production-step count, distinct from trajectory ``step``.
NPT additionally records ``ensemble=npt_scr``, ``condition_label`` and
``target_pressure`` in GPa. Box entries and atom data use 17 significant digits;
velocity uses the same Angstrom/fs units as NVT restart XYZ files. All required
metadata, condition assignments, box constraints and random states are checked.
