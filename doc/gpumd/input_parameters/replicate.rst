.. _kw_replicate:
.. index::
   single: replicate (keyword in run.in)

:attr:`replicate`
=================

Replicate the simulation cell along its three cell vectors.

Syntax
------

::

  replicate <n_a> <n_b> <n_c>

Parameters
----------

Here, :attr:`n_a`, :attr:`n_b` and :attr:`n_c` are the numbers of replicas in the :math:`a`, :math:`b` and :math:`c` directions. They must be positive integers.


Examples
--------

Build a :math:`1 \times 2 \times 4` supercell::

   replicate 1 2 4

Notes
-----
* Use this command before loading the potential.
* The group information and velocities of atoms are also replicated.
* Do not replicate along nonperiodic directions.
