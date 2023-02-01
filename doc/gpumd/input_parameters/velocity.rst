.. _kw_velocity:
.. index::
   single: velocity (keyword in run.in)

:attr:`velocity`
================

This keyword is used to initialize the velocities of the atoms in the system according to a given temperature. 

Syntax
------
* This keyword only has one parameter, which is the initial temperature of the system::
  
    velocity <initial_temperature>

* The temperature is in units of kelvin (K).

Example
-------

The command::

    velocity 10

means that one wants to set the initial temperature to 10 K. 

Caveats
-------

* The initial velocities generated in this fashion do not obey the Maxwell distribution.
  This is, however, not a problem since during the MD simulation, the Maxwell distribution will be achieved automatically within a short time.
* The total linear and angular momenta are set to zero.
* If there are already velocity data in the :ref:`simulation model file <model_xyz>`, the initial temperature will not be used and the velocities will be initialized as those in the simulation model file.
  **Important:** In this case, you still need to write this keyword, otherwise the velocities will not be initialized at all (and will hence be undefined).
