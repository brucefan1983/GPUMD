.. _kw_potential:
.. index::
   single: potential (keyword in run.in)

:attr:`potential`
=================

This keyword is used to specify the interatomic potential model for the system.

Available potential models
--------------------------

* :ref:`Tersoff-1989 potential <tersoff_1989>`
* :ref:`Tersoff-1988 potential <tersoff_1988>`
* :ref:`Tersoff mini potential <tersoff_mini>`
* :ref:`Embedded atom method (EAM) potential <eam>`
* :ref:`Force constant potential (FCP) <fcp>`
* :ref:`Lennard-Jones (LJ) potential <lennard_jones_potential>`
* :ref:`Neuroevolution potential (NEP) <nep_formalism>`
* :ref:`Hybrid NEP+ILP potential <nep_ilp>`
* :ref:`Hybrid SW+ILP potential <sw_ilp>`
* :ref:`Hybrid Tersoff+ILP potential <tersoff_ilp>`
* :ref:`Deep Potential (DP) <use_dp_in_gpumd>`

Syntax
------

The first parameter, :attr:`potential_filename`, is the filename (including
relative or absolute path) of the potential file to be used. Some potential
types require additional parameters, as described in their corresponding
sections.

For standard NEP, temperature-dependent NEP, and ``nep_spin`` models, a multi-GPU run can
optionally specify the spatial partition direction::

  potential <potential_filename> [partition_direction]

Here, :attr:`partition_direction` can be ``x``, ``y``, or ``z``. These
correspond to the ``a``, ``b``, and ``c`` directions, respectively, for a
triclinic box. If the direction is omitted, GPUMD selects it automatically,
normally along the thickest direction of the box.

The optional partition direction applies only to the multi-GPU implementation
of standard NEP, temperature-dependent NEP, and ``nep_spin`` models. Charge NEP models
currently use the single-GPU implementation in ``gpumd``.

Example
-------

.. code::

   potential Si_NEP.txt

For a multi-GPU NEP run, one can force the decomposition along a specific
direction, for example::

   potential Si_NEP.txt x

NEP-Spin spatial decomposition
------------------------------

A ``nep_spin`` simulation uses all GPUs visible to the process for the same
atomic system. For example, request four GPUs from the scheduler and run::

   CUDA_VISIBLE_DEVICES=0,1,2,3 ./gpumd

The model and the TSPIN/SIB ensemble commands are unchanged. The optional
partition direction above also applies to NEP-Spin. Fully periodic orthogonal
and triclinic cells are supported, as in the single-GPU Spin3 implementation.
SOC, noncollinear spins, and group/region ``add_mfield`` use the same physical
conventions as the single-GPU path. This is single-process, single-node MD;
it does not distribute training configurations or independent trajectories.

Positions, types, and spins are repacked on every force evaluation, including
repeated evaluations at unchanged atomic positions. Ownership and neighbour lists can be reused while every
atom moves less than 0.2 times the maximum cutoff from the partition reference.
A five-cell halo (2.5 cutoffs) covers the two-hop environment plus this motion;
exceeding the displacement bound or changing the cell rebuilds the partition. Each spatial slab includes at least two maximum cutoffs
of halo atoms: the neighbours of an owned atom must themselves have complete
local environments for the force and magnetic-force derivatives. The existing
Spin3 evaluator computes each extended slab. Only owned-atom energy, force,
magnetic force, and all nine virial components are returned to GPU 0; ghost
outputs are discarded rather than added a second time. The external field and
integration act on the assembled global state.

Small cells may require the entire geometry on each active GPU and therefore
need not speed up. Slabs have equal geometric widths; a strongly nonuniform
atomic distribution can cause load imbalance. Integration and global atom
arrays remain on GPU 0, so memory capacity does not grow exactly in proportion
to the GPU count. Prefer a long, evenly populated partition direction and
GPUs connected by fast peer links. Runtime-staged copies are supported when
direct peer access is unavailable, but can reduce throughput.
