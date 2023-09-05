.. _kw_minimize:
.. index::
   single: minimize (keyword in run.in)

:attr:`minimize`
================

This keyword is used to minimize the energy of the system.
Currently, the fast inertial relaxation engine (FIRE) [Bitzek2006]_ [Guénolé2020]_ method and the steepest descent (SD) method has been implemented.


Syntax
------

This keyword is used as follows::

  minimize <method> <force_tolerance> <maximal_number_of_steps>

Here,
:attr:`method` can be :attr:`sd` (the steepest descent method) or :attr:`fire` (the FIRE method).
:attr:`force_tolerance` is in units of eV/Å.
When the largest absolute force component among the :math:`3N` force components in the system is smaller than :attr:`force_tolerance`, the energy minimization process will stop even though the number of steps (interations) performed is smaller than :attr:`maximal_number_of_steps`.
:attr:`maximal_number_of_steps` is the maximal number of steps (interations) to be performed for the energy minimization process.

Examples
--------

Example 1
^^^^^^^^^
The command::

  minimize sd 1.0e-6 10000

means that one wants to do an energy minimization using the steepest descent method, with a force tolerance of :math:`10^{-6}` eV/Å for up to 10,000 steps.

Example 2
^^^^^^^^^
If you have no idea how small :attr:`force_tolerance` should be, you can simply asign a negative number to it::

  minimize sd -1 10000

In this case, the energy minimization process will definitely run 10,000 steps.

Example 3
^^^^^^^^^
The command::

  minimize fire 1.0e-5 1000

means that one wants to do an energy minimization using the steepest descent method, with a force tolerance of :math:`10^{-5}` eV/Å for up to 1,000 steps.

Caveats
-------

* This keyword should occur after the :ref:`potential keyword <kw_potential>`.
* Currently, the simulation box is fixed during the energy minimization.
