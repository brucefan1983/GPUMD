.. _kw_compute_cmass:
.. index::
   single: compute_cmass (keyword)

``compute_cmass``
=================

Writes the Cartesian center of mass to :ref:`cmass.out <cmass_out>`.

Syntax
------

.. code-block::

   compute_cmass <sample_interval> [group <grouping_method> <group_id>]

The sampling interval is a positive integer in MD steps.
The optional ``group`` arguments select a nonempty group using its zero-based grouping method and group ID.
Omit ``group`` to compute the center of mass of all atoms.

The center of mass is :math:`\sum_i m_i \mathbf{r}_i / \sum_i m_i`, using the simulation's per-atom masses and unwrapped Cartesian positions.
The positions follow each atom across periodic boundaries, so the reported center of mass can lie outside the simulation box.
For a trajectory with matching positions, use :ref:`dump_xyz <kw_dump_xyz>` with ``unwrapped_position_only``.
When a new GPUMD process starts from :ref:`restart.xyz <restart_xyz>`, periodic image tracking starts from the saved coordinates.

For example, the following command samples group 0 of grouping method 0 every 10 MD steps.

.. code-block::

   compute_cmass 10 group 0 0
