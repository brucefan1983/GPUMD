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
  
  compute_msd <sample_interval> <Nc> [<option> ...]

with parameters defined as

* :attr:`sample_interval`: Sampling interval of the position data
* :attr:`Nc`: Maximum number of correlation steps

The optional arguments can be given in any order. Each option can be specified at most once.
The available options are described below.
The first special keyword is :attr:`group`.
The parameters are:

* :attr:`group <group_method> <group>`, where :attr:`group_method` is the grouping method to use for computation and :attr:`group` is the group in the grouping method to use

The second special keyword is :attr:`all_groups`.
This keyword computes the :term:`MSD` and :term:`SDC` for each group in the specified grouping method.
Note that :attr:`group` and :attr:`all_groups` are mutually exclusive.
A typical usecase could be to compute the :term:`MSD` for each molecule in a system.
The parameters are:

* :attr:`all_groups <group_method>`, where :attr:`group_method` is the grouping method to use for computation

Finally, the third special keyword is :attr:`save_every`.
This keyword saves the internal :term:`MSD` and :term:`SDC` computed so far during the simulation, which can be helpful during long running simulations.
The file will have a name formatted as ``msd_step[step].out``. 
The parameters are:

* :attr:`save_every <interval>`, where :attr:`interval` is a positive integer giving the number of steps between saving a copy. Note that the copy can only be written at most every :attr:`sample_interval` steps. Furthermore, the first ``msd_step[step].out`` file will be written after :attr:`Nc` times :attr:`sample_interval` steps. Subsequent files will be written every :attr:`interval`.


Examples
--------

An example of this function is::

  compute_msd 5 200 group 1 1

This means that you

* want to calculate the :term:`MSD`
* the position data will be recorded every 5 steps
* the maximum number of correlation steps is 200
* you would like to compute only over group 1 in group method 1.


To compute the :term:`MSD` for all groups in group method 1 and save a copy of the :term:`MSD` every 100 000 steps, one can write::

  compute_msd 5 200 all_groups 1 save_every 100000
