.. _observer_out:
.. index::
   single: observer.out (output file)

``observer.out``
================

This file contains the global thermodynamic quantities sampled at a given frequency, for each of the specified potentials.
This file is generated when the :ref:`dump_observer keyword <kw_dump_observer>` is invoked, which also controls the frequency of the output.

* If `mode` in `dump_observer` is set to `observe`, one file is written for each of the `N` specified potentials, named `observer0.out`, `observer1.out`, ..., `observer(N-1).out` in the order of the potentials in `run.in`.
  With a single potential, the file is named `observer.out`.
* If `mode` in `dump_observer` is set to `average`, a single file named `observer.out` is written, holding the thermodynamic quantities computed with the average potential.

Refer to :ref:`thermo.out <thermo_out>` for the format of this file.
Under PIMD in `observe` mode, a row holds the temperature and kinetic energy of the centroid velocities, and the potential energy and pressure of the potential at the centroid.
Under PIMD in `average` mode, a row equals the row of `thermo.out`, with the target temperature and the quantum kinetic energy.

