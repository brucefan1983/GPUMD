.. _kw_compute_hac:
.. index::
   single: compute_hac (keyword in run.in)

:attr:`compute_hac`
===================

This keyword can be used to calculate the heat current autocorrelation (:term:`HAC`) and running thermal conductivity (:term:`RTC`) using the :ref:`Green-Kubo method <green_kubo_method>`.

Syntax
------

::

  compute_hac <sampling_interval> <correlation_steps> <output_interval>

Parameters
----------

The first parameter is the sampling interval for the heat current data.
The second parameter is the maximum number of correlation steps.
The third parameter is the output interval of the :term:`HAC` and :term:`RTC` data.

Output
------

The results will be written to the :ref:`hac.out output file <hac_out>`.

Examples
--------

Example 1
^^^^^^^^^

.. code::

   time_step 1
   compute_hac 10 100000 1
   run 10000000

This 10 ns run samples the heat current every 10 fs, giving one million samples.
The correlation window spans approximately 1 ns.
With an output interval of 1, no adjacent correlation points are averaged before writing, giving 100000 rows.

Example 2
^^^^^^^^^

.. code::

   compute_hac 10 100000 10

This is similar to the above example but with one exception:
The :term:`HAC`/:term:`RTC` data will be averaged for every 10 data before outputting, generating :math:`10^4` rows of data in the output file.

Notes
-----

None.
