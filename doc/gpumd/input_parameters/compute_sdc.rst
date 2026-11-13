.. _kw_compute_sdc:
.. index::
   single: compute_sdc (keyword in run.in)

``compute_sdc``
===============

Compute the velocity autocorrelation (:term:`VAC`) and integrate it to obtain the self-diffusion coefficient (:term:`SDC`).
Results are written to :ref:`sdc.out <sdc_out>`.

Syntax
------

::

  compute_sdc <sample_interval> <Nc> [group <group_method> <group>]

Parameters
----------

* ``sample_interval``: Positive integer giving the number of MD steps between velocity samples.
* ``Nc``: Positive integer giving the number of correlation points, including zero correlation time.
* ``group <group_method> <group>``: Optional selection of a group defined in :ref:`model.xyz <grouping_methods>`.
  Without this option, all atoms are included.

Examples
--------

Sample velocities every 5 steps and calculate 200 correlation points for group 1 of grouping method 1::

  compute_sdc 5 200 group 1 1

Notes
-----

The run must provide at least ``Nc`` velocity samples.
Repeat this command before each run that requires the calculation; each run accumulates its own averages.
