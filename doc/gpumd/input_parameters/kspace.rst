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
(default: 1.0; allowed range: 0.2 to 2.0, inclusive).
Smaller values give finer meshes at a higher computational cost.

The mesh is chosen automatically and grows as needed when the box changes.
It does not shrink. The actual spacing may be smaller than the requested value.

Specify ``kspace`` at most once in ``run.in``, before the first ``run``.

Example
-------

To use Ewald::

   kspace ewald

To request a PPPM spacing of at most 1.5 Angstrom use::

   kspace pppm 1.5
