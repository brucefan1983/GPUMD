.. _kw_potential:
.. index::
   single: potential (keyword in run.in)

:attr:`potential`
=================

This keyword is used to specify the interatomic potential model for the system.

The :ref:`interatomic potentials chapter <potentials>` describes model forms and parameter files.
External integrations are described under :ref:`Deep Potential <use_dp_in_gpumd>` and :ref:`NNAP <use_nnap_in_gpumd>`.

Syntax
------

::

  potential <potential_filename> [<model_options>]

Parameters
----------

The first parameter, :attr:`potential_filename`, is the filename (including
relative or absolute path) of the potential file to be used. Some potential
types require additional parameters, as described in their corresponding
sections.

NEP models must use the :ref:`nep.txt format <nep_txt>` with
``format_version 1`` and one hidden layer.

For standard NEP, temperature-dependent NEP, and ``nep_spin`` models, a multi-GPU run can
optionally specify the spatial partition direction::

  potential <potential_filename> [partition_direction]

Here, :attr:`partition_direction` can be ``x``, ``y``, or ``z``. These
correspond to the ``a``, ``b``, and ``c`` directions, respectively, for a
triclinic box. If the direction is omitted, GPUMD selects it automatically,
normally along the thickest direction of the box.

The optional partition direction applies only to the multi-GPU implementation
of standard NEP, temperature-dependent NEP, and ``nep_spin`` models. For ordinary MD, charge and vdW NEP models
currently use the single-GPU implementation in ``gpumd``; see :ref:`Long-range models <nep_long_range_models>` for their boundary conditions and usage.
With :ref:`multi_replica <kw_multi_replica>`, complete replicas are assigned to GPUs and the partition-direction parameter is not supported.

Examples
--------

.. code::

   potential Si_NEP.txt

For a multi-GPU NEP run, one can force the decomposition along a specific
direction, for example::

   potential Si_NEP.txt x

Notes
-----

For NEP-spin input requirements, integrator choices, output, and continuation, see :ref:`NEP-spin runtime use <nep_spin_runtime>`.
