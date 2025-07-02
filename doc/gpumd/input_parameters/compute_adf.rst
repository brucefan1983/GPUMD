.. _kw_compute_adf:
.. index::
   single: compute_adf (keyword in run.in)

:attr:`compute_adf`
===================

This keyword computes the angular distribution function (:term:`ADF`) for all atoms or specific triples of species. Each :term:`ADF` is represented as a histogram, created by measuring the angles formed between a central atom and two neighboring atoms, and binning these angles into `num_bins` bins. Only neighbors with distances `rc_min < R < rc_max` are considered, where `rc_min` and `rc_max` are specified separately for the first and second neighbor atoms in each :term:`ADF` calculation.
Currently, this feature is only available for classical :term:`MD`.
The results are written to the :ref:`adf.out <adf_out>` file.

Syntax
------

For global :term:`ADF`, the keyword is used as follows::

  compute_adf <interval> <num_bins> <rc_min> <rc_max>

This means the :term:`ADF` calculations will be performed every :attr:`interval` steps, with :attr:`num_bins` data points.

For local :term:`ADF`, the keyword is used as follows::

  compute_adf <interval> <num_bins> <itype1> <jtype1> <ktype1> <rc_min_j1> <rc_max_j1> <rc_min_k1> <rc_max_k1> ...

This means the :term:`ADF` calculations will be performed every :attr:`interval` steps, with :attr:`num_bins` data points. 
The angle formed by the central atom I and neighboring atoms J and K is included in the :term:`ADF` if the following conditions are met:

- The distance between atoms I and J is between `rc_min_jN` and `rc_max_jN`.
- The distance between atoms I and K is between `rc_min_kN` and `rc_max_kN`.
- The type of atom I matches `itypeN`.
- The type of atom J matches `jtypeN`.
- The type of atom K matches `ktypeN`.
- Atoms I, J, and K are distinct.

The :term:`ADF` value for a bin is computed by dividing the histogram count by the total number of triples that satisfy the criteria, ensuring that the integral of the :term:`ADF` with respect to the angle is 1. In other words, the :term:`ADF` is a probability density function.

Example
-------

  - compute_adf 100 30 0.0 1.2  # Total :term:`ADF` every 100 MD steps with 30 data points for bond lengths between 0.0 and 1.2
  - compute_adf 500 50 0 1 1 0.0 1.2 0.0 1.3  # Calculate 0-1-1 :term:`ADF` every 500 MD steps with 50 data points for I-J bond lengths between 0.0 and 1.2, and I-K bond lengths between 0.0 and 1.3
  - compute_adf 500 50 0 1 1 0.0 1.2 0.0 1.3 1 0 1 0.0 1.2 0.0 1.3  # Calculate 0-1-1 and 1-0-1 :term:`ADF` every 500 MD steps with 50 data points
