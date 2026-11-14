.. _observer_xyz:
.. index::
   single: observer.xyz (output file)

``observer.xyz``
================

File containing atomistic positions, velocities and forces.
It is generated when invoking the :ref:`dump_observer keyword <kw_dump_observer>`.

* In the case of `dump_observer` being in `observe` mode, an XYZ-file for each potential is written with the index of the potential in `run.in` appended to the file name.
  For example, `observer0.xyz` corresponds to the first NEP potential specified in `run.in`, `observer1.xyz` to the second, and so forth.
  With a single potential, the file is named `observer.xyz`.
* In the case of `mode` being `average`, a single file named `observer.xyz` is written, corresponding to the average of the supplied NEP potentials.

File format
-----------
This file is in the `extended XYZ format <https://github.com/libAtoms/extxyz>`_.
The output mode for this file is append.
