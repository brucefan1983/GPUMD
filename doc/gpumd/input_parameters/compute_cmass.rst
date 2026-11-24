.. _kw_compute_cmass:
.. index::
   single: compute_cmass (keyword in run.in)

``compute_cmass``
=====================

This keyword computes the center of mass from the per-atom masses and unwrapped Cartesian positions.
It supports classical molecular dynamics only.
GPUMD reports an input error for PIMD-family ensembles, including RPMD and TRPMD.

Syntax
------

::

  compute_cmass <sample_interval> [group <group_method> <group>]

Parameters
----------

* ``sample_interval`` is a positive integer giving the number of MD steps between samples.
* ``group <group_method> <group>`` selects one nonempty group by its zero-based grouping method and group ID.
  Omit ``group`` to include all atoms.

Output
------

The results are written to the :ref:`cmass.out output file <cmass_out>`.

Examples
--------

To sample group 0 of grouping method 0 every 10 MD steps, use::

  compute_cmass 10 group 0 0

Notes
-----

The center of mass is :math:`\sum_i m_i \mathbf{r}_i / \sum_i m_i`.
The unwrapped positions follow each atom across periodic boundaries, so the center of mass can lie outside the simulation box.
Tracking starts when GPUMD first enables periodic image tracking. Crossings in earlier untracked runs cannot be recovered.
For the absolute center of mass of a molecule or cluster, use initial coordinates that represent a contiguous object. Per-atom unwrapping does not reassemble an object that starts split across a periodic boundary.

Specify ``compute_cmass`` before each ``run`` that should produce output.
Periodic image tracking persists across consecutive classical runs in the same GPUMD process.
For a trajectory with matching positions, use :ref:`dump_xyz <kw_dump_xyz>` with ``unwrapped_position_only``.
When a new GPUMD process starts from :ref:`restart.xyz <restart_xyz>`, periodic image tracking starts from the saved coordinates.
