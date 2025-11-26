.. _kw_compute_dpdt:
.. index::
   single: compute_dpdt (keyword in run.in)

:attr:`compute_dpdt`
====================

This keyword can be used to calculate the time derivative of the polarizability via the dot product between the Born effective charge (:term:`BEC`) and the velocity. 
The keyword is only meaningful for a simulation with the qNEP model trained with target :term:`BEC` values.
The results will be written to the file :ref:`dpdt.out <dpdt_out>`.

Syntax
------
This keyword has 1 parameter::

  compute_dpdt <sampling_interval>

Here,

* :attr:`sampling_interval` is the sampling interval for calculating the time derivative of the polarizability.

Examples
--------

Example 1
^^^^^^^^^

.. code::

   compute_dpdt 5

This means to calculate the time derivative of the polarizability every 5 time steps.