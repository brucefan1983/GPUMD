.. _kw_dump_beads:
.. index::
   single: dump_beads (keyword in run.in)

:attr:`dump_beads`
==================

Write some data (positions and optionally velocities and forces) into `beads_dump_<k>.xyz` in `extended XYZ format <https://github.com/libAtoms/extxyz>`_, where `<k>` runs from 0 to `number_of_beads - 1` as set in a run related to path-integral molecular dynamics (:term:`PIMD`).
Each file thus contains the data for one bead.
The comment line of each frame carries the time, the periodic boundary flags, and the cell.

Caveats
-------

* The centroid data (the average over all the beads) for the ring polymer in PIMD-related runs are still output by the :ref:`dump_xyz keyword <kw_dump_xyz>`.

* This keyword should not be used in a run not related to PIMD.
