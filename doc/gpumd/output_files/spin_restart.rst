.. _spin_restart:

Spin restart
============

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
