.. _kw_kspace:
.. index::
   single: kspace (keyword in run.in)

:attr:`kspace`
==============

This keyword selects the method for computing the reciprocal-space electrostatic energy.

Syntax
------

::

  kspace ewald
  kspace pppm [spacing]

The default method is ``pppm`` (particle-particle particle-mesh).
Its optional ``spacing`` parameter sets the maximum mesh spacing in Angstrom
(allowed range: 0.2 to 2.0, inclusive).
Smaller values give finer meshes at a higher computational cost.

The default spacing is :math:`\min(1, r_c^R/6)` Angstrom,
where :math:`r_c^R` is the radial cutoff of the model in Angstrom.
The Ewald splitting parameter is :math:`\alpha = \pi/r_c^R`.
The PPPM error grows with the product of :math:`\alpha` and the spacing.
The default spacing keeps this product at or below :math:`\pi/6`.

The mesh is chosen automatically and grows as needed when the box changes.
It does not shrink. The actual spacing may be smaller than the requested value.

At the first force evaluation, GPUMD prints the mesh
and an estimate of the root-mean-square error of the PPPM forces relative to the Ewald sum [Deserno1998]_.
The estimate uses the charges of that step and assumes that they are randomly distributed.
For a crystal, the actual error can be smaller by an order of magnitude or more.

Specify ``kspace`` at most once in ``run.in``.
The setting applies to every ``run``, wherever the line appears in ``run.in``.

Example
-------

To use Ewald::

   kspace ewald

To request a PPPM spacing of at most 1.5 Angstrom use::

   kspace pppm 1.5
