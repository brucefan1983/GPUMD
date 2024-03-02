.. _kw_compute_sdc:
.. index::
   single: compute_sdc (keyword in run.in)

:attr:`compute_sdc`
===================

This keyword computes the self-diffusion coefficient (:term:`SDC`) from the velocity autocorrelation (:term:`VAC`) function.
If this keyword appears in a run, the :term:`VAC` function will be computed and integrated to obtain the :term:`SDC`.
The results will be written to :ref:`sdc.out output file <sdc_out>`.

Syntax
------
For this keyword, the command looks like::
  
  compute_sdc <sample_interval> <Nc> [<optional_arg>]

with parameters defined as

* :attr:`sample_interval`: Sampling interval of the velocity data
* :attr:`Nc`: Maximum number of correlation steps

The optional argument :attr:`optional_arg` allows an additional special keyword.
The keyword for this function is :attr:`group`.
The parameters are:

* :attr:`group <group_method> <group>`, where :attr:`group_method` is the grouping method to use for computation and :attr:`group` is the group in the grouping method to use

Examples
--------

An example of this function is::

  compute_sdc 5 200 group 1 1

This means that you

* want to calculate the :term:`SDC`
* the velocity data will be recorded every 5 steps
* the maximum number of correlation steps is 200
* you would like to compute only over group 1 in group method 1.

Caveats
-------
This function cannot be used in the same run with the :ref:`compute_dos keyword <kw_compute_dos>`.
