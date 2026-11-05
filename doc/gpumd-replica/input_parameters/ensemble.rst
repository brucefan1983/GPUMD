.. _replica_kw_ensemble:
.. index::
   single: ensemble (keyword in gpumd_replica run.in)

:attr:`!ensemble`
=================

Syntax
------

::

  ensemble nvt_bdp <T_start> <T_end> <T_coup>

Only ``nvt_bdp`` with a fixed simulation box is supported.
It uses the stochastic velocity-rescaling thermostat of Bussi, Donadio, and Parrinello [Bussi2007b]_.
Temperatures are in K and must be positive.
``T_coup`` is the thermostat relaxation time divided by the time step and must be at least 1.

* REMD: the temperatures must match the ladder endpoints; they do not define a heating ramp.
* PRD: both temperatures must be equal.

Example
-------

For REMD between 300 and 450 K::

  ensemble nvt_bdp 300 450 100

Caveats
-------

The thermostat uses 3N degrees of freedom for REMD and 3N-3 for PRD.
PRD removes center-of-mass motion and requires at least two atoms.
