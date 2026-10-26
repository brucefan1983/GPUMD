.. _kw_dump_restart:
.. index::
   single: dump_restart (keyword in run.in)

:attr:`dump_restart`
====================

Write a restart file.

Syntax
------

This keyword only requires a single parameter, which is the output interval (number of steps) of updating the restart file::

  dump_restart <interval>

Example
-------

To update the restart file every 100000 steps for a run, one can add::

  dump_restart 100000

before the :ref:`run keyword <kw_run>`.


Continuing a Spin NVT run
-------------------------

For ``nvt_glsd`` with lattice motion enabled (the default), copy the new
``restart.xyz`` to ``model.xyz`` in a new directory and keep the original
potential, timestep, temperature endpoints, coupling, spin parameters and
seed. Do not regenerate velocities. For example, after step 400 of an
original ``run 1000`` with a 300 to 500 K ramp::

  potential nep.txt
  ensemble nvt_glsd 300 500 100 damping 0.01 seed 731
  time_step 0.1
  dump_restart 100
  run 600

Repeat the original endpoints rather than starting a new ramp at the
checkpoint temperature. The saved protocol phase applies to both the atomic
thermostat and the default spin bath. An explicitly specified positive
``stemp`` remains constant. Beyond the original protocol length, targets hold
their final values. Constant-temperature runs may simply be extended.

Spin NVT ensembles with lattice motion enabled write versioned ``nhc_state``
and ``nhc_time`` metadata, with all four
chain masses, positions and conjugate momenta, original protocol length,
completed steps and global time. Values use 17 significant digits. Chain
masses retain the existing convention of initialization at the original
starting temperature; they are not recalculated at the resumed temperature.
Configuration or timestep mismatches are rejected. Old files without these
fields still load but initialize a new thermostat. Remove both fields only
when intentionally starting a new bath protocol.

``lattice off`` has no atomic NHC and writes no NHC metadata. Selecting it or
another driver discards pending NHC state with a message. NHC and MTTK state
must not coexist in one input. Ordinary ``nvt_nhc``, ``heat_nhc`` and
``heat_nhc_power`` retain the upstream restart behavior and do not write or
restore this Spin lattice checkpoint. The same model, fixed-timestep and stateful-
modifier limitations described below apply. Ordinary SIB random-stream
continuation is not provided by saving its atomic thermostat alone.


Continuing a Spin NPT run
-------------------------

Copy ``restart.xyz`` to ``model.xyz`` in a new run directory. Specify the
same potential, atom order, timestep and ensemble settings. Do not regenerate
velocities. For example, after a checkpoint at step 400 of a planned 1000-step
run, continue with the remaining 600 steps::

  potential nep.txt
  ensemble npt_glsd temp 300 500 iso 0 1 tperiod 100 pperiod 1000 damping 0.01 seed 731
  time_step 0.1
  dump_restart 100
  run 600

The original run must have used the same temperature/pressure endpoints and
``run 1000``. Repeat those original endpoints, not the temperature/pressure
at the checkpoint. The checkpoint retains the original ramp length and
completed step count. After its original ramp length is reached, further
steps hold the final targets. Constant-target runs may simply be extended.
This does not change the usual behavior of separate ``run`` commands without
loading a checkpoint.

MTTK checkpoints contain versioned ``mttk_state`` and ``mttk_time`` metadata
in the XYZ header. They save both Nose--Hoover chain states, cell-rate tensor
and masses, the non-hydrostatic reference cell, cached targets and the protocol
phase. Atomic positions, velocities and cell retain double-precision text;
GLSD also saves full-precision moments and its noise counter/seed. Restore
checks the MTTK configuration, atom count and timestep instead of silently
resetting a mismatched bath. Do not edit the serialized state.

Files from earlier versions without MTTK metadata remain readable, but start
new thermostat/barostat states and cannot reproduce an uninterrupted NPT
trajectory. Selecting a different, non-MTTK ensemble discards the MTTK state
with a message. Removing both MTTK metadata fields intentionally starts a new
bath protocol; this is not an exact continuation.

The NPT support applies to the Spin MTTK lattice driver in ``npt_glsd``,
``npt_sib`` and ``npt_tspin``. Ordinary ``nvt_mttk``, ``npt_mttk`` and
``nph_mttk`` retain the upstream behavior and do not write or restore these
endpoint fields. Loading a Spin checkpoint into an ordinary ensemble discards
its lattice bath state and starts the execution clock from zero.
It does not checkpoint every other thermostat or stateful modifier in GPUMD.
Use fixed timesteps and the same mechanical/spin fields and model. Separate
random mechanical forces, velocity-reset schedules, temperature-dependent NEP
protocols, analysis histories and unwrapped-position image counters are not
covered by the thermostat checkpoint contract. ``nvt_glsd`` uses the separate
NHC driver described above. SIB's random-stream
continuation is also outside this GLSD checkpoint guarantee.

Floating-point force reductions and neighbor ordering can differ after
rebuilding a simulation, especially across GPU counts. Expect agreement to
validated numerical tolerances, not bitwise identical long trajectories.


Caveats
-------
This keyword is not propagating.
That means, its effect will not be passed from one run to the next.
