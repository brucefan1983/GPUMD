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

  compute_cohesive <e1> <e2> <direction>

Here,
:attr:`e1` is the smaller box-scaling factor,
:attr:`e2` is the larger box-scaling factor, and
:attr:`direction` specifies the direction of the scaling, and can be either ``0 ~ 6``, which correspond to the x, y, z, xy, yz, zx, and xyz directions, respectively.


Examples
--------

The command::

  compute_cohesive 0.9 1.2 0

means that one wants to compute the cohesive energy curve in the x-direction from the box-scaling factor 0.9 to the box-scaling factor 1.2, with ``(e2 - e1)*1000 +1`` points.
The box-scaling points will be 0.9, 0.901, 0.902, ..., 1.2.


Caveats
-------

This keyword must occur after the :ref:`potential keyword <kw_potential>`.
