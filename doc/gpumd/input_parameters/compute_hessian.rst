.. _kw_compute_hessian:

.. index::
   single: compute_hessian (keyword in run.in)

:attr:`compute_hessian`
=======================

This keyword computes the analytic Cartesian Hessian matrix of the current
configuration with a :term:`NEP` potential. It is an immediate, standalone
command: it is executed as soon as it is read and does not require an
:ref:`ensemble <kw_ensemble>` or a :ref:`run <kw_run>` command.

Unlike :ref:`compute_phonon <kw_compute_phonon>`, which uses the
finite-displacement method, :attr:`compute_hessian` evaluates the second
derivatives of the potential energy analytically. If the analytic provider does
not support the model or the configuration, the command reports an error and
terminates; it never silently falls back to a finite-displacement result.

Syntax
------

.. code::

   compute_hessian method analytic [key value ...]

The command must appear after the :ref:`potential <kw_potential>` definition.
Only ``method analytic`` is accepted. All further arguments are optional
key-value pairs; each key is followed by exactly one value.

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
     - The NEP potential file references are read from this file
   * - ``kpoints.in``
     - Optional; required only for ``phonon dispersion`` (see :ref:`kpoints_in`)

Output files
------------

The analytic Hessian is written to ``hessian.out`` (or to the path given by
``output``). Optional outputs are only created when explicitly requested:

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: auto

   * - Output filename
     - Brief description
   * - ``hessian.out``
     - Symmetrized analytic Hessian matrix (default; rename with ``output``)
   * - ``raw_output`` path
     - Un-symmetrized analytic Hessian matrix (optional, see ``raw_output``)
   * - ``element_errors.csv``
     - Elementwise comparison with a finite-displacement reference (optional)
   * - ``D.out``
     - Dynamical matrices for the requested :math:`\boldsymbol{k}`-points
   * - ``omega2.out``
     - Squared phonon frequencies :math:`\omega^2` in inverse picoseconds squared
   * - ``eigenvector.out``
     - Eigenvalues and eigenvectors for a sole Gamma point, binary format

The format of ``D.out`` and ``omega2.out`` produced by this command follows the
corresponding :ref:`D.out <D_out>` and :ref:`omega2.out <omega2_out>` output
files, with the differences noted in the section :ref:`hessian_phonons` below.

Parameters
----------

``output <path>``
   Path of the symmetrized analytic matrix. Default: ``hessian.out``.

``output_format dense|matrix_market``
   Choose the dense text matrix (``dense``, the default) or a sparse Matrix
   Market coordinate matrix (``matrix_market``). The sparse path assembles the
   NEP Hessian contributions directly into a symmetric :math:`3\times3`
   block-compressed row workspace and writes only the nonzero scalar elements in
   the SoA row and column order. It currently supports a single ordinary NEP
   model without DFT-D3 or interlayer-potential (:term:`ILP`) environments.
   Sparse output does not support ``raw_output``, finite-difference validation,
   or Hessian metadata.

``raw_output <path>``
   Path of the un-symmetrized analytic matrix. Disabled by default.

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

``validate_fd yes|no``
   Compare the analytic matrix with a central finite-difference reference that
   is built from force evaluations with displacement ``displacement``. Default:
   ``no``. This is a comparison, not a fallback mechanism.

``displacement <value>``
   Positive finite Cartesian displacement in Å used for the finite-difference
   reference. Default: ``0.001``.

``fd_output <path>``
   Write the finite-difference reference matrix. Setting this option implies
   ``validate_fd yes``.

``element_errors <path>``
   Path of the elementwise comparison between the analytic matrix and the
   finite-difference reference. Default: ``element_errors.csv``. The comparison
   requires a reference, so use it together with ``validate_fd`` or
   ``fd_output``; setting this option alone does not enable finite differences.

``phonon none|gamma|dispersion``
   Optionally solve the phonon problem from the same analytic Hessian. The
   default, ``none``, only writes the matrix.

``supercell nx,ny,nz``
   Repeat counts of the current structure relative to the reference cell, given
   as three comma-separated positive integers. Default: ``1,1,1``. Requires a
   phonon mode. The counts describe an already existing structure; this option
   does not replicate atoms.

``kpoints <path>``
   Path to the high-symmetry path file. Default: ``kpoints.in``. Only accepted
   together with ``phonon dispersion``.

``kpoint_intervals <integer>``
   Number of interpolation intervals per path segment. Default: ``100``. Only
   accepted together with ``phonon dispersion``.

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
mass-weighted dynamical matrix. The symmetrized matrix is written to ``output``;
the un-symmetrized matrix is written to ``raw_output`` when requested.

Every output file starts with comment lines recording the coordinate order,
matrix order, definition, units, number of atoms and the kind of matrix.

Analytic support and limitations
--------------------------------

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
and terminates. Finite-difference fallback is disabled: ``validate_fd`` is a
comparison, not a substitute for an unsupported analytic Hessian.

Existing output files from earlier runs are not removed when an error occurs.
Use a clean output directory to avoid confusing old results with the results of
a failed run.

Before writing, all enabled output paths are checked against each other and
against ``model.xyz``, ``run.in``, the active kpoints file, and the potential
files referenced in ``run.in``. Absolute paths, parent-directory aliases,
symbolic links (including dangling output links) and hard links are resolved for
this check, and conflicts are rejected. This is a preflight check; it does not
protect against another process changing a path after validation, and output
publication is not transactional.

.. _hessian_phonons:

Phonons from the analytic Hessian
---------------------------------

``phonon gamma`` forms a mass-weighted real dynamical matrix and computes all
eigenvalues and eigenvectors. By default the entire current structure is used as
the basis, which works for both molecules and periodic supercell Gamma modes
without :ref:`replicate <kw_replicate>` or ``kpoints.in``. Rigid modes are not
projected out and negative eigenvalues are retained. Optimize the structure
before interpreting the frequencies.

``phonon dispersion`` Fourier transforms the analytic force constants and
solves the complex Hermitian eigenproblem at every requested wavevector.
Fractional wavevectors refer to the reciprocal lattice of the reference cell
defined by ``supercell``. Each non-comment line of the kpoints file has four
fields::

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

The interaction range estimate is conservative: for NEP a force-constant
interaction range of twice the potential cutoff is used. This is an
interpolation guard, not an additional truncation of the analytic Hessian.

The full :math:`3N\times3N` analytic Hessian is still computed and stored, so
reduced phonon matrices do not remove this quadratic memory requirement.
:math:`\boldsymbol{k}`-points are processed one at a time to avoid storing all
dynamical matrices simultaneously.

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

The default command writes the symmetrized analytic Hessian to ``hessian.out``::

   potential /absolute/path/nep.txt
   compute_hessian method analytic

Compare with a central finite-difference reference and write the raw matrix,
the reference matrix and the elementwise errors::

   potential /absolute/path/nep.txt
   compute_hessian method analytic raw_output raw.txt validate_fd yes displacement 0.001 fd_output fd.txt element_errors errors.csv

Gamma modes of the entire current system::

   potential /absolute/path/nep.txt
   compute_hessian method analytic phonon gamma

Dispersion of an explicitly replicated reference crystal::

   replicate 6 6 6
   potential /absolute/path/nep.txt
   compute_hessian method analytic phonon dispersion supercell 6,6,6 kpoints kpoints.in kpoint_intervals 100

The repeat counts above are an example, not a universally sufficient cell size.
The original finite-displacement :ref:`compute_phonon <kw_compute_phonon>`
command is unchanged and remains available.

To also produce a structure snapshot for external mode analysis, use::

   compute_hessian method analytic structure_output hessian_structure.xyz

Caveats
-------

This keyword must occur after the :ref:`potential <kw_potential>` definition.

The command only supports the analytic NEP path described above. In particular,
the standard finite-displacement :ref:`compute_phonon <kw_compute_phonon>`
command should be used when an analytic Hessian is unavailable.

The cost and memory of the analytic Hessian scale quadratically with the number
of atoms :math:`N`, since the full :math:`3N\times3N` matrix is stored. This is
especially important for ``phonon dispersion``, where the requested supercell
must be large enough to satisfy the interaction-range condition above.

For a molecule, a valid Hessian should have exactly six zero modes (three
translations and three rotations) if the structure is fully relaxed and the
frequencies are computed for the isolated system. For a periodic crystal, the
three acoustic modes should vanish at Gamma. Deviations from these limits
indicate that the structure is not at a stationary point or that the supercell
and cutoff parameters are insufficient.
