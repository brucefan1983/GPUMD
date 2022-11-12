.. _kw_compute_cohesive:
.. index::
   single: compute_cohesive (keyword in run.in)

:attr:`compute_cohesive`
========================

This keyword is used for computing the cohesive energy curve.
The results are written to the :ref:`cohesive.out output file <cohesive_out>`.


Syntax
------

This keyword is used as follows::

  compute_cohesive e1 e2 num_points

Here,
:attr:`e1` is the smaller box-scaling factor,
:attr:`e2` is the larger box-scaling factor, and
:attr:`num_points` is the number of points sampled uniformly from :attr:`e1` to :attr:`e2`.


Examples
--------

The command::

  compute_cohesive 0.9 1.2 301

means that one wants to compute the cohesive energy curve from the box-scaling factor 0.9 to the box-scaling factor 1.2, with 301 points.
The box-scaling points will be 0.9, 0.901, 0.902, ..., 1.2.


Caveats
-------

This keyword must occur after the :ref:`potential keyword <kw_potential>`.
