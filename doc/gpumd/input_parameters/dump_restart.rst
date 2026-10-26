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


Caveats
-------
This keyword is not propagating.
That means, its effect will not be passed from one run to the next.

Spin lattice checkpoints
------------------------

Spin NVT/NPT runs save the atomic thermostat/barostat state and original ramp
phase as paired ``nhc_state``/``nhc_time`` or ``mttk_state``/``mttk_time`` fields.
Copy ``restart.xyz`` to ``model.xyz`` and retain the potential, atom order,
timestep, temperature/pressure endpoints and ensemble parameters. Do not
regenerate velocities. The original ramp continues and holds its final target
beyond its saved length. Mismatched configurations are rejected; selecting a
different lattice driver discards pending state. ``lattice off`` has no lattice
thermostat checkpoint. Ordinary NHC/MTTK ensembles retain upstream behavior.

This preserves the lattice bath, not every extended variable or random stream
of the spin integrator. TSPIN's spin thermostat state is not checkpointed.
Other stateful modifiers and analysis histories are outside this contract.
