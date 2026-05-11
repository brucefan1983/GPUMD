.. _kw_compute_ic:
.. index::
   single: compute_ic (keyword in run.in)

:attr:`compute_ic`
===================

This keyword computes the iron conductivity (:term:`IC`) from the mean-square displacement (:term:`MSD`) function.
If this keyword appears in a run, the :term:`IC` function will be calculated as a time derivative of it.
The results will be written to :ref:`ic.out output file <ic_out>`.

Syntax
------
For this keyword, the command looks like::
  
  compute_ic <sample_interval> <Nc> <type_iex> <charge>

with parameters defined as

* :attr:`sample_interval`: Sampling interval of the position data
* :attr:`Nc`: Maximum number of correlation steps
* :attr:`type_iex`: Type index of atom to be calculated which in potential
* :attr:`charge`: Charge of the ion to compute the conductivity.


Examples
--------

An example of this function is::

  compute_ic 5 200 0 1

This means that you

* want to calculate the iron conductivity from the mean-square displacement function
* the position data will be recorded every 5 steps
* the maximum number of correlation steps is 200
* you would like to compute only for the atom with type 0 in nep.txt and charge 1.

