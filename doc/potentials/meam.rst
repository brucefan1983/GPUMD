.. _meam_potential:

Modified embedded-atom method
=============================

CUDA and HIP builds support the standard modified embedded-atom method (MEAM)
with multi-element parameters and the second-nearest-neighbor extension.
MEAM reads the elemental library and optional override file used by
`LAMMPS pair_style meam <https://docs.lammps.org/pair_meam.html>`_ in metal
units (eV and Å). Set the interaction cutoff in the override file. To select
the second-nearest-neighbor formulation for element pair I,J, set
``nn2(I,J) = 1``. A zero or omitted setting selects the original formulation.

Input file
----------

Save the three-line descriptor as ``meam.txt`` for this example. If the parameter file
uses indices 1 and 2 for Ga and In, respectively, write::

   meam 2 Ga In
   library.meam
   GaIn.meam

List the elements in the same order used to index the parameter file.
Use ``NULL`` on the third line when using library defaults. Relative paths are
resolved from the descriptor's directory. The optional fourth line
``library_elements`` provides one alternate library entry name per element.
For example, append ``library_elements Ni4`` to a descriptor with the header
``meam 1 Ni`` to select the ``Ni4`` parameterization for Ni.

Then select the descriptor in ``run.in``::

   potential meam.txt

Behavior
--------

MEAM evaluates the screened pair energy and the embedding energy of each
site. It computes analytic forces and all nine components of the GPUMD
per-atom virial. Screening candidates may lie beyond the radial pair cutoff.
Periodic box thicknesses must exceed twice the screening search radius
derived from the parameter file. Replicate smaller periodic cells before use.

The ``meam`` selector supports the standard LAMMPS library and override
format with up to 32 selected entries. Multi-state MEAM and spline MEAM have
different functional forms and file formats and require separate selectors.

Reference
---------

M. I. Baskes, Phys. Rev. B **46**, 2727 (1992),
`doi:10.1103/PhysRevB.46.2727 <https://doi.org/10.1103/PhysRevB.46.2727>`_.

B.-J. Lee and M. I. Baskes, Phys. Rev. B **62**, 8564 (2000),
`doi:10.1103/PhysRevB.62.8564 <https://doi.org/10.1103/PhysRevB.62.8564>`_.
