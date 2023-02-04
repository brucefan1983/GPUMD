.. index::
   single: correct_velocity (keyword in run.in)

:attr:`correct_velocity`
========================

This keyword allows one to enforce zero angular momentum.
This can be useful as the stochastic nature of most thermostats and barostats available in :program:`gpumd` can cause the conservation of angular momentum to be violated.

Syntax
------
* This keyword only has one parameter, which is the interval between which the angular momentum is set to zero::
  
    correct_velocity <interval>


Example
-------

The command::

    correct_velocity 10

implies that the angular momentum is strictly set to zero every 10 steps.
