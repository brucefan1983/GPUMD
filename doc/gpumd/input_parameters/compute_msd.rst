.. _kw_compute_msd:
.. index::
   single: compute_msd (keyword in run.in)

:attr:`compute_msd`
===================

This keyword computes the self-diffusion coefficient (:term:`SDC`) from the mean-square displacement (:term:`MSD`) function.
If this keyword appears in a run, the :term:`MSD` function will be computed and each Cartesian component of the :term:`SDC` is calculated as one half of the time derivative of the corresponding :term:`MSD` component.
The results will be written to :ref:`msd.out output file <msd_out>`.

Syntax
------
::

  compute_msd <sample_interval> <Nc> [<option> ...]

Parameters
----------

* :attr:`sample_interval`: Positive integer giving the number of time steps between position samples
* :attr:`Nc`: Positive integer giving the number of correlation points, including zero correlation time

Each optional argument can appear once, in any order:

* ``group <group_method> <group>`` selects one group.
* ``all_groups <group_method>`` computes separate results for every group in the selected grouping method, for example one group per molecule.
  It cannot be combined with ``group``.
* ``save_every <interval>`` writes intermediate results to ``msd_step[step].out``.
  ``interval`` is a positive integer in MD time steps.

A copy is written only on a step that is a multiple of both :attr:`sample_interval` and :attr:`interval`, after at least :attr:`Nc` position samples have been collected.
The first copy is therefore written at the first such step at or after ``Nc * sample_interval``.
Choose :attr:`interval` to be a multiple of :attr:`sample_interval` to obtain the requested spacing between saved copies.
Step numbers are counted from the beginning of the current run.
Each copy contains the averages accumulated so far; saving does not reset the accumulation.


Examples
--------

An example of this function is::

  compute_msd 5 200 group 1 1

This samples positions every 5 steps for group 1 in grouping method 1, using 200 correlation points including zero time.

To compute the :term:`MSD` for all groups in group method 1 and save a copy of the :term:`MSD` every 100 000 steps, one can write::

  compute_msd 5 200 all_groups 1 save_every 100000

In this example, sufficient position samples are available at step 1000, but the first intermediate file is ``msd_step100000.out`` at step 100000.
The final results are also appended to ``msd.out`` at the end of the run.
