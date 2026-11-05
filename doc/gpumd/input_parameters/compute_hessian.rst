.. _kw_compute_hessian:

.. index::
   single: compute_hessian (keyword in run.in)

:attr:`compute_hessian`
=======================

The :attr:`compute_hessian` keyword computes the Cartesian Hessian matrix of
the current configuration and can optionally solve phonons from it. It is an
immediate, standalone command: it is executed when it is read and does not
require an :ref:`ensemble <kw_ensemble>` or a :ref:`run <kw_run>` command.

Two Hessian modes are available:

* ``compute_hessian analytic`` computes the Hessian analytically. This mode is
  currently available for supported :term:`NEP` models only.
* ``compute_hessian fd`` computes the Hessian by central finite differences of
  the forces. This mode supports every potential implemented by the
  :ref:`potential <kw_potential>` command, including non-NEP potentials and
  composite potentials such as those containing an interlayer potential
  (:term:`ILP`).

If the analytic mode is not supported for the selected potential, model, or
configuration, it reports an error and terminates. It does not automatically
fall back to the finite-difference mode; use ``compute_hessian fd`` instead.

Syntax
------

.. code::

   compute_hessian analytic [key value ...]
   compute_hessian fd       [key value ...]

The command must appear after the :ref:`potential <kw_potential>` definition.
The first argument after the keyword selects the mode. All further arguments
are optional key-value pairs; each key is followed by exactly one value.

The following parameters are available in both modes:

``output <path>``
   Path of the symmetrized Hessian matrix. Default: ``hessian.out``.

``structure_output <path>``
   Write an extended XYZ snapshot of the exact configuration for which the
   Hessian was evaluated, including cell, periodic boundary conditions, species,
   positions, masses and forces. The file also records the conservative
   interaction range used by the :math:`\boldsymbol{q}`-mesh postprocessing.
   Use a fresh, distinct path that does not coincide with an input file or
   another output.

``metadata <path>``
   Write a diagnostic JSON file describing the Hessian, the potential files and
   their hashes, the device, and the matrix diagnostics. Disabled by default.

``phonon none|gamma|dispersion``
   Optionally solve phonons from the Hessian. The default is ``none``.
   ``gamma`` computes the Gamma-point modes of the current structure.
   ``dispersion`` computes modes along a path read from ``kpoints`` and uses
   ``supercell`` to identify the reference cell. See
   :ref:`hessian_phonons` below.

``supercell nx,ny,nz``
   Number of reference-cell repeats along the three lattice directions. The
   default is ``1,1,1``. This parameter is accepted only together with
   ``phonon gamma`` or ``phonon dispersion``.

``kpoints <path>``
   Path to the high-symmetry path file. Default: ``kpoints.in``. This parameter
   is accepted only together with ``phonon dispersion``.

``kpoint_intervals <integer>``
   Number of interpolation intervals per path segment. Default: ``100``. This
   parameter is accepted only together with ``phonon dispersion``.

The following parameters are available only in analytic mode:

``output_format dense|sparse``
   Choose the dense text matrix (``dense``, the default) or a sparse Matrix
   Market coordinate matrix (``sparse``). Sparse output is available
   only with ``compute_hessian analytic`` for a single supported NEP model and
   is described in :ref:`hessian_sparse`. It is incompatible with
   ``raw_output`` and ``metadata``.

``raw_output <path>``
   Path of the un-symmetrized analytic matrix. Disabled by default. This
   parameter is available only in analytic mode.

The following parameter is available only in finite-difference mode:

``displacement <value>``
   Positive finite Cartesian displacement in Å used by the central-difference
   formula. Default: ``0.001``. This parameter is available only in
   ``compute_hessian fd`` mode.

Input files
-----------

The command uses the following files from the input directory:

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: auto

   * - Input filename
     - Brief description
   * - ``model.xyz``
     - Simulation model containing the configuration to be evaluated
   * - ``run.in``
     - The potential file references are read from this file
   * - ``kpoints.in``
     - Optional; required only for ``phonon dispersion`` (see :ref:`kpoints_in`)

Output files
------------

The symmetrized Hessian is written to ``hessian.out`` (or to the path given by
``output``). Optional outputs are only created when explicitly requested:

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: auto

   * - Output filename
     - Brief description
   * - ``hessian.out``
     - Symmetrized Hessian matrix (default; rename with ``output``)
   * - ``raw_output`` path
     - Un-symmetrized analytic Hessian matrix (analytic mode only)
   * - ``metadata`` path
     - Diagnostic JSON metadata (disabled by default)
   * - ``structure_output`` path
     - Extended XYZ snapshot of the evaluated configuration
   * - ``D.out``
     - Dynamical matrices for the requested :math:`\boldsymbol{k}`-points
   * - ``omega2.out``
     - Squared phonon frequencies :math:`\omega^2` in inverse picoseconds squared
   * - ``eigenvector.out``
     - Eigenvalues and eigenvectors for a sole Gamma point, binary format

The format of ``D.out`` and ``omega2.out`` produced by this command follows the
corresponding :ref:`D.out <D_out>` and :ref:`omega2.out <omega2_out>` output
files, with the differences noted in the section :ref:`hessian_phonons` below.

Hessian matrix
--------------

The matrix has shape :math:`3N\times3N`, where :math:`N` is the number of atoms,
and is stored in the SoA ordering

.. math::

   x_0,\dots,x_{N-1},y_0,\dots,y_{N-1},z_0,\dots,z_{N-1}

on both rows and columns. It is defined as the negative Jacobian of the forces
with respect to the Cartesian coordinates,

.. math::

   H_{ij} = -\frac{\partial F_i}{\partial r_j}
          = \frac{\partial^2 E}{\partial r_i \partial r_j},

and is written in units of eV/Å\ :sup:`2`. This is a Cartesian Hessian, not a
mass-weighted dynamical matrix.

The matrix written to ``output`` has been symmetrized as
:math:`(H+H^\mathsf{T})/2`. In analytic mode the un-symmetrized matrix is
written to ``raw_output`` when requested. In finite-difference mode the
unavoidable small asymmetry caused by numerical force evaluations is removed by
the same symmetrization before ``output`` is written; no un-symmetrized fd
matrix is written.

Every dense output file starts with comment lines recording the coordinate
order, matrix order, definition, units, number of atoms and the kind of matrix.
For finite-difference output, the result is identified as
``finite_difference_symmetrized``; for analytic output it is identified as
``analytic_symmetrized`` (or ``analytic_raw`` for ``raw_output``).

.. _hessian_sparse:

Sparse Hessian output
---------------------

``compute_hessian analytic output_format sparse`` avoids constructing
the dense :math:`3N\times3N` matrix during the Hessian calculation. It builds
the symmetric block sparsity pattern of the NEP Hessian from the radial and
angular neighbor lists, accumulates each :math:`3\times3` Cartesian
atom-pair block directly in a compressed row workspace on the device, and then
symmetrizes the blocks as :math:`(H+H^\mathsf{T})/2`. The stored workspace
scales with the number of populated atom-pair blocks rather than with
:math:`N^2`.

The selected ``output`` path (by default, ``hessian.out``) is written as a
Matrix Market coordinate file:

.. code::

   %%MatrixMarket matrix coordinate real general
   % coordinate_order=soa
   % definition=minus_force_jacobian
   % unit=eV/A^2
   % atom_block_size=3
   % N=N
   3N 3N nnz
   row column value

The dimensions are :math:`3N\times3N`, and ``nnz`` is the number of nonzero
scalar entries actually written. Only nonzero entries inside the populated
:math:`3\times3` blocks are stored. The Matrix Market indices are one-based
and follow the same SoA ordering as the dense output,

.. math::

   \mathrm{row} = a_i N + i + 1, \qquad
   \mathrm{column} = a_j N + j + 1,

where :math:`a_i,a_j\in\{0,1,2\}` select the :math:`x,y,z` Cartesian
component. Both symmetric entries :math:`(i,j)` and :math:`(j,i)` are written,
and the values have units of eV/Å\ :sup:`2`. The file is therefore a general
Matrix Market matrix rather than a symmetric-storage matrix, and it can be read
by standard Matrix Market readers.

Sparse output is supported only for the analytic Hessian of a single NEP4
energy model without a DFT-D3 or interlayer-potential (:term:`ILP`) component;
the other restrictions of analytic mode also apply. ``raw_output`` and
``metadata`` cannot be combined with ``output_format sparse``.
``structure_output`` remains available. If a ``phonon`` mode is requested, the
sparse blocks are expanded to a dense matrix for the phonon solver, so the
sparse representation reduces matrix storage only for Hessian-only
calculations.

.. _hessian_analytic:

Analytic mode
-------------

The generic keyword does not imply analytic support for every potential. The
current analytic provider supports only a single NEP energy model: the ordinary
or temperature-dependent NEP4 variants, with an angular expansion of
:math:`L_\mathrm{max}\le 8` and at most 80 angular terms. It does not support:

* a standalone non-NEP potential such as Lennard-Jones or Tersoff;
* composite models that combine a NEP with an interlayer potential (:term:`ILP`)
  or DFT-D3, even if those components are supported elsewhere in
  :program:`GPUMD`;
* more than one potential, including the ``observe`` and ``average``
  multiple-potential modes used by :ref:`dump_observer <kw_dump_observer>`;
* model or configuration parameters outside the ranges checked by the provider.

If the analytic provider rejects the model or configuration, or fails the
built-in symmetry and translation-sum validation, the command reports an error
and terminates. Finite-difference fallback is not performed; use
``compute_hessian fd`` when an analytic Hessian is unavailable.

Finite-difference mode
----------------------

For each Cartesian coordinate :math:`r_j`, the finite-difference mode evaluates
the forces at :math:`r_j+\delta` and :math:`r_j-\delta`, where :math:`\delta` is
the value of ``displacement``. It then forms every column of the Hessian using
the central-difference formula

.. math::

   H_{ij} = -\frac{F_i(r_j+\delta)-F_i(r_j-\delta)}{2\delta}.

This requires :math:`2\times3N` force evaluations in addition to the baseline
evaluation at the unperturbed configuration. The resulting matrix is
symmetrized before it is written to ``output``. A smaller ``displacement`` can
reduce truncation error but may increase floating-point noise in the force
differences; a larger value may introduce finite-displacement truncation error.
The default is a compromise appropriate for many force fields, but the optimal
value can depend on the potential and configuration.

The finite-difference mode can be used with any force field supported by
GPUMD. In particular, it is the recommended mode for non-NEP potentials,
multiple potentials, and models containing an ILP or DFT-D3 component. The
interaction range used by phonon postprocessing is taken as twice the largest
potential cutoff, which is a conservative estimate for composite potentials.

.. _hessian_phonons:

Phonons from the Hessian
------------------------

Both Hessian modes can optionally produce phonons. ``phonon gamma`` forms a
mass-weighted real dynamical matrix and computes all eigenvalues and
eigenvectors. By default the entire current structure is used as the basis,
which works for both molecules and periodic supercell Gamma modes without
:ref:`replicate <kw_replicate>` or ``kpoints.in``. Rigid modes are not projected
out and negative eigenvalues are retained. Optimize the structure before
interpreting the frequencies.

``phonon dispersion`` Fourier transforms the force constants and solves the
complex Hermitian eigenproblem at every requested wavevector. Fractional
wavevectors refer to the reciprocal lattice of the reference cell defined by
``supercell``. Each non-comment line of the kpoints file has four fields::

   0.0 0.0 0.0 G
   0.5 0.0 0.0 X

Blank lines separate disconnected paths; their start points are retained at the
same accumulated path distance as the preceding endpoint. Comment-only lines do
not break a path. An empty file, a malformed point, or a nonzero wavevector
component in a nonperiodic direction is rejected.

For a reduced basis, the atom ordering must match the :program:`GPUMD`
replication order: reference-cell atoms fastest, then :math:`z` copies,
:math:`y` copies and :math:`x` copies. Positions, types, masses and group labels
must repeat consistently. Equivalent cell origins are averaged before the
Fourier transformation. No empirical force-constant truncation, acoustic
sum-rule correction, or nonanalytic polar correction is imposed.

Periodic-image information is already folded into a finite-supercell Hessian.
For noncommensurate wavevectors the periodic cell thickness must exceed twice
the estimated force-constant interaction range; otherwise the command rejects
the request instead of incorrectly interpolating aliased image contributions.
Commensurate points, for which ``q[d]*supercell[d]`` is an integer along a
periodic direction, do not require image separation. This condition is checked
at every interpolated point, not only at the path endpoints. Use a larger
supercell for a continuous dispersion.

The interaction range estimate is conservative: a force-constant interaction
range of twice the largest potential cutoff is used. This is an interpolation
guard, not an additional truncation of the Hessian.

For dense Hessian output, the full :math:`3N\times3N` matrix is stored, so
reduced phonon matrices do not remove this quadratic memory requirement. With
``output_format sparse``, requesting phonons expands the sparse blocks
into a dense matrix before solving; the sparse representation reduces matrix
storage only when no phonon mode is requested. :math:`\boldsymbol{k}`-points
are processed one at a time to avoid storing all dynamical matrices
simultaneously.

Phonon outputs
--------------

When a phonon mode is requested, ``hessian.out`` is still written, and the
following files are produced:

* ``D.out``: dynamical matrices stacked by :math:`\boldsymbol{k}`-point.
  Rows and columns are atom-major within the reference basis, and the entries are
  in native inverse-time-squared units. A sole Gamma point has :math:`3B` real
  columns; otherwise each row has :math:`3B` real columns followed by
  :math:`3B` imaginary columns. This differs from the SoA Cartesian Hessian
  layout.
* ``omega2.out``: comment headers followed by the path distance in
  Å\ :sup:`-1` and :math:`3B` ascending squared angular frequencies in
  inverse picoseconds squared, where :math:`B` is the number of atoms in the
  reference basis. Ordinary frequencies in THz are
  :math:`\sqrt{\omega^2}/(2\pi)` for positive eigenvalues. Negative values
  indicate negative curvature and are not clipped.
* ``eigenvector.out``: written only for a sole Gamma point, in the legacy binary
  format: :math:`3B` single-precision squared angular frequencies followed by
  :math:`3B` eigenvectors, each containing :math:`3B` single-precision
  components in SoA order. These are normalized mass-weighted eigenvectors, not
  Cartesian displacements.

The output file names above are reserved when phonons are enabled. Old files
from previous runs are not deleted; in particular, a dispersion calculation does
not remove an old ``eigenvector.out``. Use separate output directories and check
the process exit status before using any file from a failed run.

Examples
--------

The default analytic command writes the symmetrized analytic Hessian to
``hessian.out``::

   potential /absolute/path/nep.txt
   compute_hessian analytic

Write both the raw and symmetrized analytic matrices::

   potential /absolute/path/nep.txt
   compute_hessian analytic raw_output raw.txt

Write the symmetrized analytic Hessian in sparse Matrix Market format::

   potential /absolute/path/nep.txt
   compute_hessian analytic output_format sparse output hessian.mtx

Compute a central finite-difference Hessian with the default displacement::

   potential /absolute/path/lj.txt
   compute_hessian fd

Use a larger finite-difference displacement::

   compute_hessian fd displacement 0.01

Compute Gamma modes using either Hessian mode::

   compute_hessian analytic phonon gamma
   compute_hessian fd phonon gamma

Compute a dispersion for an explicitly replicated reference crystal::

   replicate 6 6 6
   potential /absolute/path/nep.txt
   compute_hessian analytic phonon dispersion supercell 6,6,6 kpoints kpoints.in kpoint_intervals 100

The repeat counts above are an example, not a universally sufficient cell size.
The original finite-displacement :ref:`compute_phonon <kw_compute_phonon>`
command is unchanged and remains available.

To also produce a structure snapshot for external mode analysis, use::

   compute_hessian analytic structure_output hessian_structure.xyz

Caveats
-------

This keyword must occur after the :ref:`potential <kw_potential>` definition.

The memory required for dense output and for phonon postprocessing scales
quadratically with the number of atoms :math:`N`, since the full
:math:`3N\times3N` matrix is stored. ``output_format sparse`` instead
scales with the number of populated atom-pair blocks for Hessian-only
calculations, but it is expanded to a dense matrix whenever phonons are
requested. The quadratic memory requirement is especially important for
``phonon dispersion``, where the requested supercell must be large enough to
satisfy the interaction-range condition above. The analytic mode computes one
Hessian; the finite-difference mode additionally requires :math:`2\times3N`
force evaluations and can therefore be considerably more expensive for large
systems.

For a molecule, a valid Hessian should have exactly six zero modes (three
translations and three rotations) if the structure is fully relaxed and the
frequencies are computed for the isolated system. For a periodic crystal, the
three acoustic modes should vanish at Gamma. Deviations from these limits
indicate that the structure is not at a stationary point, that the
finite-difference displacement is not suitable, or that the supercell and cutoff
parameters are insufficient.
