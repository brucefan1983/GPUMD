.. _spin_restart:

Spin restart
============

Use ``dump_restart <interval>`` to save the current cell and per-atom state.
To start another simulation, copy ``restart.xyz`` to ``model.xyz``, provide the
potential and the desired ``run.in``, and run GPUMD normally. Do not generate
new atomic velocities if you want to reuse the saved values. For example::

  potential nep.txt
  ensemble nvt_glsd 300 300 100 damping 0.01 seed 913
  time_step 0.1
  dump_restart 1000
  run 10000

The file contains positions, masses, atomic velocities, groups and magnetic
moments. When TSPIN spin velocities are initialized, it also contains
``spin_vel:R:3``. TSPIN reuses these velocities; when they are absent, it
initializes them from the input temperature and seed. SIB and GLSD have no
independent spin velocity.

As with ordinary GPUMD restart, this is a physical-state snapshot.
NHC/MTTK internal variables, the execution clock, temperature/pressure ramp
progress, and random-number state are not saved. Thermostats and barostats
are initialized from the new input, and ramps span the newly requested run.
Choose the initial temperature/pressure for the stage you want to simulate.
The resulting trajectory need not match an uninterrupted simulation.

Each SIB or GLSD ``run`` initializes its noise sequence from ``seed``.
Reusing the same seed repeats the sequence from its beginning; choose another
seed when a different sequence is desired. Within the same GPUMD process,
atomic and spin coordinates remain current across run boundaries, and TSPIN
also reuses current spin velocities. Writing restart output does not update
or control the integrator's physical state.
