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

For standard NEP and temperature-dependent NEP models, a multi-GPU run can
optionally specify the spatial partition direction::

  potential <potential_filename> [partition_direction]

Here, :attr:`partition_direction` can be ``x``, ``y``, or ``z``. These
correspond to the ``a``, ``b``, and ``c`` directions, respectively, for a
triclinic box. If the direction is omitted, GPUMD selects it automatically,
normally along the thickest direction of the box.

The optional partition direction applies only to the multi-GPU implementation
of standard NEP and temperature-dependent NEP models. Charge NEP models
currently use the single-GPU implementation in ``gpumd``.

Example
-------

.. code::

   potential Si_NEP.txt

For a multi-GPU NEP run, one can force the decomposition along a specific
direction, for example::

   potential Si_NEP.txt x
