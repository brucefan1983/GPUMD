.. index::
   single: correct_velocity (keyword in run.in)

:attr:`correct_velocity`
========================

This keyword allows one to enforce zero linear and angular momenta at regular intervals.

Syntax
------
* This keyword can have one or two parameters, which are the interval between which the linear and angular momenta are set to zero and an optional group method::
  
    correct_velocity <interval> [<group_method>]

* :attr:`interval` must be larger than or equal to 10. A value between 10 and 100 should be good.

* The :attr:`group_method` must be defined in :attr:`model.xyz`.

Example 1
---------

The command::

    correct_velocity 10

implies that the linear and angular momenta are set to zero every 10 steps.

Example 2
---------

The command::

    correct_velocity 50 3

implies that the linear and angular momenta for the individual groups in group method 3 are set to zero every 50 steps.
