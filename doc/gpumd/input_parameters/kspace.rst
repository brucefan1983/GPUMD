.. _kw_kspace:
.. index::
   single: kspace (keyword in run.in)

:attr:`kspace`
==============

This keyword is used to set the computation method for the reciprocal space contribution to the electrostatic energy.

Syntax
------

This keyword is used as follows::

  kspace ewald
  kspace pppm [spacing]

The default method is ``pppm`` (particle-particle particle-mesh).
The optional ``spacing`` is a finite positive number in Angstrom, with a
default of 1.0. It sets an upper bound on the mesh-plane spacing in each
direction. A smaller value gives a finer mesh at a higher computational cost.
``ewald`` does not accept a spacing parameter.

For direction ``d``, the box thickness is the volume divided by the area of
the opposite face. This definition also applies to triclinic boxes.
The required mesh count is first rounded up from ``thickness / spacing``.
It is then increased to the smallest even integer with only the prime factors
2, 3, 5 and 7, with a minimum of 16 in each direction.
The actual spacing can therefore be smaller than the requested value.

During a simulation, a mesh direction grows whenever needed to preserve the
spacing bound. It never shrinks, including across successive ``run`` commands
using the same potential object. Box contraction thus reduces the actual
spacing without rebuilding the mesh. Box-dependent reciprocal-space factors
are updated at every force evaluation, even when the mesh counts stay fixed.
The initial mesh and its actual spacing are printed once, at the first force
evaluation. Later mesh growth is silent.

``kspace`` is a global setting and may appear only once in ``run.in``.
It is recommended to place it after the potential definitions and before the
first run. The existing input-snapshot behavior also recognizes a later
setting. The setting is not changed between runs.

This spacing control applies to the main ``gpumd`` executable.

Example
-------

To use the Ewald method use::

   kspace ewald

To request a PPPM spacing of at most 1.5 Angstrom use::

   kspace pppm 1.5

Omitting ``kspace``, writing ``kspace pppm``, and writing
``kspace pppm 1.0`` select the same default settings.
