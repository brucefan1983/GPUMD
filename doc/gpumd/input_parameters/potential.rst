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

NEP-spin
--------

A :ref:`NEP-spin model <nep_spin>` uses all GPUs visible to the process to
compute the same atomic system. The optional partition direction also applies
to NEP-spin. Small systems may not benefit from multiple GPUs; global atom
arrays remain on GPU 0.

Provide ``spin:R:3`` in :ref:`model.xyz <model_xyz>`. Fully periodic orthogonal
and triclinic cells are supported. Spins remain fixed under ``nve`` and
``nvt_nhc``; other ensembles and multiple-potential runs are not supported.
See :ref:`the Spin runtime instructions <nep_spin_runtime>` for output,
restart and external-field usage.
