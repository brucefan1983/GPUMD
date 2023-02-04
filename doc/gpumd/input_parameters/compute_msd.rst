.. _kw_compute_msd:
.. index::
   single: compute_msd (keyword in run.in)

:attr:`compute_msd`
===================

This keyword computes the self-diffusion coefficient (:term:`SDC`) from the mean-square displacement (:term:`MSD`) function.
If this keyword appears in a run, the :term:`MSD` function will be computed and the :term:`SDC` is also calculated as a time derivative of it.
The results will be written to :ref:`msd.out output file <msd_out>`.

Syntax
------
For this keyword, the command looks like::
  
  compute_msd <sample_interval> <Nc> [<optional_arg>]

with parameters defined as

* :attr:`sample_interval`: Sampling interval of the position data
* :attr:`Nc`: Maximum number of correlation steps

The optional argument :attr:`optional_arg` allows an additional special keyword.
The keyword for this function is :attr:`group`.
The parameters are:

* :attr:`group <group_method> <group>`, where :attr:`group_method` is the grouping method to use for computation and :attr:`group` is the group in the grouping method to use

Examples
--------

An example of this function is::

  compute_msd 5 200 group 1 1

This means that you

* want to calculate the :term:`MSD`
* the position data will be recorded every 5 steps
* the maximum number of correlation steps is 200
* you would like to compute only over group 1 in group method 1.
