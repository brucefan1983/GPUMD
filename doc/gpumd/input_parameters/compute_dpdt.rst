.. _kw_compute_dpdt:
.. index::
   single: compute_dpdt (keyword in run.in)

:attr:`compute_dpdt`
====================

This keyword calculates the time derivative of the total dipole moment of the simulation cell by contracting each atomic Born effective charge (:term:`BEC`) tensor with the atomic velocity and summing over atoms.
The keyword is only meaningful for a simulation with the qNEP model trained with target :term:`BEC` values.
The results will be written to the file :ref:`dpdt.out <dpdt_out>`.

Syntax
------
This keyword has 1 parameter::

  compute_dpdt <sampling_interval>

Parameters
----------

Here,

* :attr:`sampling_interval` is a positive integer giving the number of time steps between samples.

Examples
--------

Example 1
^^^^^^^^^

.. code::

   compute_dpdt 5

This means to calculate the time derivative of the total dipole moment every 5 time steps.
