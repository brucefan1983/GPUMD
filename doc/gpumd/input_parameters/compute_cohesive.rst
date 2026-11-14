.. _kw_compute_cohesive:
.. index::
   single: compute_cohesive (keyword in run.in)

:attr:`compute_cohesive`
========================

This keyword is used for computing the cohesive energy curve.
At each scaling factor, the atomic positions are relaxed in the fixed, scaled cell using steepest descent, with a force tolerance of :math:`10^{-5}` eV/Å and at most 1000 minimization steps.
The results are written to the :ref:`cohesive.out output file <cohesive_out>`.
The calculation is performed when this keyword is read.


Syntax
------

This keyword is used as follows::

  compute_cohesive <e1> <e2> <direction>

Here,
:attr:`e1` is the positive initial box-scaling factor, and
:attr:`e2` is the upper target factor, which must be larger than :attr:`e1`.
:attr:`direction` takes values from 0 to 6, selecting the cell vectors a, b, c, ab, bc, ca, and abc, respectively.
For an axis-aligned orthogonal cell, these correspond to the x, y, z, xy, yz, zx, and xyz directions.
Only ``direction=6`` scales the cell isotropically.

The scaling factors are :math:`e_1+0.001n`, with :math:`n=0,\ldots,\operatorname{round}[1000(e_2-e_1)]`.
Each point is generated from the cell and atomic positions present when the keyword is read, followed by relaxation at the fixed scaled cell.


Examples
--------

The command::

  compute_cohesive 0.9 1.2 0

computes 301 points by scaling the first cell vector (the x-direction for an axis-aligned orthogonal cell).
The box-scaling points will be 0.9, 0.901, 0.902, ..., 1.2.


Caveats
-------

This keyword must occur after the :ref:`potential keyword <kw_potential>`.
