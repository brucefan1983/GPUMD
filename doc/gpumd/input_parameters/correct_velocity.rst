.. _kw_correct_velocity:
.. index::
   single: correct_velocity (keyword in run.in)

:attr:`correct_velocity`
========================

This keyword allows one to enforce zero linear and angular momenta at regular intervals.

Syntax
------
This keyword takes a correction interval and an optional grouping method::

  correct_velocity <interval> [<group_method>]

* :attr:`interval` is an integer number of MD steps and must be at least 10. A value between 10 and 100 should be good.

* When :attr:`group_method` is omitted, the correction applies to the whole system.
  When it is specified, the correction is performed independently for every group in that :ref:`grouping method <grouping_methods>` defined in ``model.xyz``.

Examples
--------

Example 1
^^^^^^^^^

The command::

    correct_velocity 10

implies that the linear and angular momenta are set to zero every 10 steps.

Example 2
^^^^^^^^^

The command::

    correct_velocity 50 3

implies that the linear and angular momenta for the individual groups in group method 3 are set to zero every 50 steps.

Run behavior
------------

The first correction is performed before the first integration step, and subsequent corrections occur every :attr:`interval` steps.
For example, an interval of 10 gives corrections before steps 1, 11, 21, and so on.
Periodic correction is disabled by default and only applies to the current :ref:`run <kw_run>` when enabled.
Repeat this command before each subsequent run that needs it.
