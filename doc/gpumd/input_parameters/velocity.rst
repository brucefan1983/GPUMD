.. _kw_velocity:
.. index::
   single: velocity (keyword in run.in)

:attr:`velocity`
================

This keyword initializes atomic velocities at a specified temperature when ``model.xyz`` contains no velocity data.

Syntax
------

::

    velocity <initial_temperature>
    velocity <initial_temperature> seed <seed_number>

Parameters
----------

* The temperature is in units of kelvin (K).
* The optional positive integer :attr:`seed_number` makes the randomly generated velocities reproducible.

Output
------

None.

Examples
--------

Initialize velocities at 10 K::

    velocity 10

Notes
-----

* The randomly generated velocities do not initially obey the Maxwell distribution; equilibrate the system before collecting production data.
  Their total linear and angular momenta are set to zero.
* If the :ref:`simulation model file <model_xyz>` contains ``vel:R:3``, those velocities are used and the initial temperature specified by ``velocity`` is ignored.
  The target temperature specified by :ref:`ensemble <kw_ensemble>` still controls the thermostat.
* If ``model.xyz`` contains no velocity data and this keyword is omitted, velocities are initialized at a default temperature of 300 K.
* To continue from the velocities reached in a preceding run, omit ``velocity`` between runs; see :ref:`multiple runs <multiple_runs>`.
