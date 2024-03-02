.. _observer_out:
.. index::
   single: observer.out (output file)

``observer.out``
================

This file contains the global thermodynamic quantities sampled at a given frequency, for each of the specified potentials.
This file is generated when the :ref:`dump_observer keyword <kw_dump_observer>` is invoked, which also controls the frequency of the output.

* If `mode` in `dump_observer` is set to `observe`, then one file will be written for each of the `N` specified potentials, with the name `observer*.out`.
* If `mode` in `dump_observer` is set to `average`, then only a single file will be written, correspond to the thermodynamic quantities computed with the average potential.

Refer to :ref:`thermo.out <thermo_out>` for the format of this file.

