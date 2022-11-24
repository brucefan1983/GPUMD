.. _kw_compute_viscosity:
.. index::
   single: compute_viscosity (keyword in run.in)

:attr:`compute_viscosity`
=========================

This keyword can be used to calculate the stress autocorrelation function and viscosity using the Green-Kubo method.
The results will be written to the :ref:`viscosity.out output file <viscosity_out>`.

Syntax
------
This keyword has 2 parameters::

  compute_viscosity sampling_interval correlation_steps

The first parameter is the sampling interval for the stress data. 
The second parameter is the total number of correlations steps. 
