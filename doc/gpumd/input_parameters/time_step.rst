.. _kw_time_step:
.. index::
   single: time_step (keyword in run.in)

:attr:`time_step`
=================

Set the time step for integration.

Syntax
------

This keyword can accept either one or two parameters. 
If there is only one parameter, it is the time step (in units of fs) for a run::

  time_step dt_in_fs

If there are two parameters, the first one is the time step and the second one is the maximum distance (in units of Ångstrom) any atom in the system can travel within one step::

  time_step dt_in_fs max_distance_per_step

Examples
--------

Example 1
^^^^^^^^^

To set the time step to 0.5 fs, one can add::

  time_step 0.5

before the :ref:`run keyword <kw_run>`.

Example 2
^^^^^^^^^

To set the time step to 2.0 fs and to require that no atom in the system can travel more than 0.05 Å per step, one can add::

  time_step 2.0 0.05

before the :ref:`run keyword <kw_run>`.


Caveats
-------
* There is a default value (1 fs) for the time step.
  That means, if you forget to set one, a time step of 1 fs will be used.
* This keyword is propagating.
  That means, its effect will be passed from one run to the next.
  Most of the other keywords are non-propagating.
