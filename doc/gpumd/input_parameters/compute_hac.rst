.. _kw_compute_hac:
.. index::
   single: compute_hac (keyword in run.in)

:attr:`compute_hac`
===================

This keyword can be used to calculate the heat current autocorrelation (:term:`HAC`) and running thermal conductivity (:term:`RTC`) using the :ref:`Green-Kubo method <green_kubo_method>`.
The results will be written to the :ref:`hac.out output file <hac_out>`.

Syntax
------
This keyword has 3 parameters::

  compute_hac <sampling_interval> <correlation_steps> <output_interval>

The first parameter is the sampling interval for the heat current data. 
The second parameter is the maximum correlations steps. 
The third parameter for is the output interval of the :term:`HAC` and :term:`RTC` data.

Examples
--------

Example 1
^^^^^^^^^

.. code::

   time_step 1
   compute_hac 10 100000 1
   run 10000000

This means that

* You want to calculate the thermal conductivity using the :ref:`Green-Kubo method <green_kubo_method>` (the :term:`EMD` method) in this run, which contains 10 milillion steps with a time step of 1 fs.
* The heat current data will be recorded every 10 steps.
  Therefore, there will be 1 million heat current data in each direction.
* The maximum number of correlation steps is :math:`10^5`, which is one tenth of the number of heat current data.
  This is a very sound choice.
  The maximum correlation time will be :math:`10^5 \times 10=10^6` time steps, i.e., 1 ns.
* The :term:`HAC`/:term:`RTC` data will not be averaged before outputting, generating :math:`10^5` rows of data in the output file.

Example 2
^^^^^^^^^

.. code::

   compute_hac 10 100000 10

This is similar to the above example but with one expection:
The :term:`HAC`/:term:`RTC` data will be averaged for every 10 data before outputing, generating :math:`10^4` rows of data in the output file.

Related tutorial
----------------

The use of the :attr:`compute_hac` keyword is illustrated in the tutorial on :ref:`thermal transport from EMD simulations <tutorials>`.
