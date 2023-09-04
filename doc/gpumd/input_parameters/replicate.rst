.. _kw_replicate:
.. index::
   single: replicate (keyword in run.in)

:attr:`replicate`
=================

Syntax
------

This keyword is used as follows::

  replicate <n_a> <n_b> <n_c>

Here, :attr:`n_a`, :attr:`n_b` and :attr:`n_c` are the numbers of replicas in the :math:`a`, :math:`b` and :math:`c` directions. They must be positive integers.


Example
-------

To build a 1*2*4 supercell, just write::

   replicate 1 2 4

Caveats
-------
* Use this command before defining potential.
* The group information and velocities of atoms are also replicated.
* Don't replicate along non-periodic dimensions.