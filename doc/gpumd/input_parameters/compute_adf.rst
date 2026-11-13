.. _kw_compute_adf:
.. index::
   single: compute_adf (keyword in run.in)

:attr:`compute_adf`
===================

This keyword computes the angular distribution function (:term:`ADF`) for all atoms or specific triples of species. Each :term:`ADF` is represented as a histogram, created by measuring the angles formed between a central atom and two neighboring atoms, and binning these angles into `num_bins` bins. Each neighbor must satisfy :math:`r_{\min} \le r < r_{\max}` for its specified radial window. The lower bound is included and the upper bound is excluded. Zero-length center-neighbor vectors are excluded because their angle is undefined.
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

- The distance between atoms I and J satisfies ``rc_min_jN <= R_IJ < rc_max_jN``.
- The distance between atoms I and K satisfies ``rc_min_kN <= R_IK < rc_max_kN``.
- The type of atom I matches `itypeN`.
- The type of atom J matches `jtypeN`.
- The type of atom K matches `ktypeN`.
- Atoms I, J, and K are distinct.

For each center I and each requested triple, an unordered pair of distinct neighbors is counted once if either assignment to J and K satisfies all the criteria.
This also applies when the two radial windows overlap: a pair that satisfies both assignments is counted only once.
The global calculation uses the same radial window for both neighbors and counts each unordered pair once.

When a multi-element NEP potential is compacted to the species required by the current simulation, `itypeN`, `jtypeN`, and `ktypeN` refer to the active atom types printed at initialization. For other potential models, the usual type numbering is unchanged.

The histogram accumulates over the sampled configurations within the current run.
For bin width :math:`\Delta\theta=180^\circ/\mathtt{num\_bins}`, the probability density is

.. math::

   p_b=\frac{H_b}{\Delta\theta\sum_c H_c},

where :math:`H_b` is the accumulated number of valid angles in bin :math:`b`.
Each global or local distribution is normalized separately, so :math:`\sum_b p_b\Delta\theta=1` when valid angles are present.
If no valid angles have been counted, the corresponding distribution is all zeros.

Examples
--------

Calculate the global ADF every 100 steps, using 30 angular bins and neighbor distances in ``[0.0, 1.2)`` Å::

   compute_adf 100 30 0.0 1.2

Calculate the local 0-1-1 ADF every 500 steps, using 50 angular bins and different radial windows for the two neighbors::

   compute_adf 500 50 0 1 1 0.0 1.2 0.0 1.3

Multiple triples can be specified in one command::

   compute_adf 500 50 0 1 1 0.0 1.2 0.0 1.3 1 0 1 0.0 1.2 0.0 1.3
