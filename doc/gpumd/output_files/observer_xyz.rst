.. _observer_xyz:
.. index::
   single: observer.xyz (output file)

``observer.xyz``
================

File containing atomistic positions, velocities and forces.
It is generated when invoking the :ref:`dump_observer keyword <kw_dump_observer>`.

* In the case of `dump_observer` being in `observe` mode, an XYZ-file for each potential will be written with the potential index in the `run.in`-file appended to the filename. For example, `observer0.xyz` corresponding to the first NEP potential specified in `run.in`, `observer1.xyz` to the second, and so forth.
* In the case of `mode` being `average`, only a single file will be written, corresponding to the average of the supplied NEP potentials. 

File format
-----------
This file is in the `extended XYZ format <https://github.com/libAtoms/extxyz>`_.
The output mode for this file is append.
