.. _kw_time_step:
.. index::
   single: time_step (keyword in run.in)

:attr:`time_step`
=================

Set the integration time step, optionally with a displacement limit for adaptive time stepping.

Syntax
------

::

  time_step <dt_in_fs> [<max_distance_per_step>]

Parameters
----------

* ``dt_in_fs``: Time step in fs. The default is 1 fs.
* ``max_distance_per_step``: Optional positive displacement limit in Å per step.
  When supplied, the time step is reduced as needed based on the largest atomic speed.
  By default, this limit is not applied.

Examples
--------

Use a fixed time step of 0.5 fs::

  time_step 0.5

Use a time step of at most 2 fs with a displacement limit of 0.05 Å per step::

  time_step 2.0 0.05

Notes
-----

* The current time-step value is retained from one run to the next unless a new ``time_step`` command is given.
* The displacement limit applies only to the current run.
  Repeat the full two-parameter command before each run that needs adaptive time stepping.
