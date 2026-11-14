.. _kw_compute_hessian:

:attr:`compute_hessian`
=======================

Compute the Cartesian Hessian analytically for a single NEP energy model at
the current atomic positions. This command writes the matrix for external
analysis. The existing :ref:`compute_phonon <kw_compute_phonon>` command
retains its finite-displacement calculation and phonon output conventions.

Syntax
------

.. code-block:: text

   compute_hessian analytic [output <path>] [output_format dense|sparse] \
       [raw_output <path>] [metadata <path>] [structure_output <path>]

Options are key-value pairs. The method must be ``analytic``. This command
must follow the :ref:`potential <kw_potential>` definition.

``output``
   Path of the symmetrized Hessian, default ``hessian.out``.

``output_format``
   ``dense`` (default) writes a full text matrix. ``sparse`` writes the
   nonzero entries in Matrix Market coordinate format, using an atom-block
   sparsity pattern built from the NEP neighbor environments.

``raw_output``
   Optional matrix before symmetrization, using the selected output format.

``metadata``
   Optional JSON file with system information, analytic energy and forces,
   and matrix diagnostics before and after symmetrization. Available with
   both dense and sparse storage.

``structure_output``
   Optional extended XYZ snapshot of the evaluated positions, masses, and
   forces. It is available with both output formats.

Output paths must differ from each other and from the input structure,
``run.in``, and potential files. Existing outputs are overwritten. File-write
failures terminate the calculation with an error.

Matrix definition and ordering
------------------------------

The Hessian is the negative Cartesian force Jacobian:

.. math::

   H_{i\alpha,j\beta}
   = -\frac{\partial F_{i\alpha}}{\partial R_{j\beta}}
   = \frac{\partial^2 E}{\partial R_{i\alpha}\partial R_{j\beta}}.

Its shape is :math:`3N\times3N` and its units are eV/angstrom squared.
Both formats use GPUMD's structure-of-arrays (SoA) coordinate order:

.. math::

   x_0,\ldots,x_{N-1},y_0,\ldots,y_{N-1},z_0,\ldots,z_{N-1}.

The matrix contains Cartesian force constants, without mass weighting or
an acoustic-sum-rule projection. The reported output is
:math:`(H+H^\mathsf{T})/2`. No dynamical matrix, wave-vector path, phonon
frequency, or eigenvector is calculated by this command.

Dense output
------------

The file begins with ``#`` comment lines identifying the ordering, units,
atom count, and whether the matrix is raw or symmetrized. Each subsequent
line is one matrix row. For example, NumPy can read it with
``numpy.loadtxt("hessian.out")``.

The raw matrix is written first if requested, then symmetrized in place.
The command does not retain a second full host matrix for symmetrization.

.. _hessian_sparse:

Sparse output
-------------

The Matrix Market file starts with::

   %%MatrixMarket matrix coordinate real general

After comments, a dimensions line gives ``3N 3N nnz``. Subsequent lines give
``row column value`` with one-based SoA indices. Both halves of the symmetric
matrix are stored; exactly zero values are omitted.

The internal pattern contains a 3-by-3 block for each pair of atoms sharing
a contributing NEP environment. Periodic images of an atom accumulate into
the same atom block. Storage scales with the number of populated blocks;
the command never expands this representation into a dense matrix.

.. _hessian_analytic:

Supported models and temperature
--------------------------------

The command uses the same ``NEP_Model_File`` reader and validation as the
NEP force evaluator. The active potential must use the single-GPU NEP
implementation; ``NEP_MULTIGPU`` and ILP combinations are rejected before
loading Hessian parameters. It accepts the current versioned model format with
``model nep`` or ``model nep_temperature``, including universal, typewise,
and flexible ZBL settings accepted by that reader. Descriptor coefficients
use the native type-pair layout, including models compacted to the species
present in ``model.xyz``. The angular expansion and optional invariants must
satisfy the shared reader's limits (currently :math:`2\le L_\mathrm{max}\le8`).

Both large boxes and periodic small boxes are supported. Small-box
calculations use the same periodic-image displacements as the force evaluator.
A single non-NEP model, multiple potentials, charge, spin, vdW, and composite
ILP models are unsupported.

DFT-D3 dispersion is explicitly rejected by the run-input dispatcher,
regardless of whether ``dftd3`` appears before or after ``compute_hessian``.
There is no finite-difference mode or fallback in this command.

For temperature-dependent models, the temperature input is fixed at the
value currently used by the force evaluator. A preceding ``run`` sets and
advances this value according to the ensemble target temperatures. Before
any ``run`` it is zero; ``velocity`` alone does not set the model temperature.
The result is the model's coordinate Hessian at that fixed temperature,
without finite-temperature sampling or phonon renormalization.

Diagnostics and metadata
------------------------

The raw matrix must be finite and satisfy these relative checks:

* maximum asymmetry divided by the largest absolute matrix entry:
  at most :math:`10^{-5}`;
* Frobenius norm of :math:`H-H^\mathsf{T}` divided by that of :math:`H`:
  at most :math:`10^{-6}`;
* largest absolute Cartesian translation sum divided by the largest
  absolute matrix entry: at most :math:`10^{-5}`.

Failed checks terminate the command. If the raw matrix is finite but fails
symmetry or translation checks, explicitly requested ``raw_output`` and
``metadata`` files are written with ``failed_validation`` status before
termination. The normal matrix and structure outputs are not written in this
case; metadata records no normal matrix output or symmetrized diagnostics.
These structural checks do not replace comparison with an independently
evaluated force Jacobian.

Metadata schema version 5 records the cell, periodicity, atom types and order,
matrix conventions and storage format, energy, forces, actual model temperature
(for temperature-dependent NEP), validation status, output paths, and
raw/symmetrized matrix diagnostics. Git information and input-file hashes are intentionally left
to external provenance or validation tools.

Examples
--------

Write a dense matrix, its raw form, and diagnostics::

   potential nep.txt
   compute_hessian analytic output hessian.out raw_output hessian_raw.out metadata hessian.json

Write a sparse matrix and the evaluated structure::

   potential nep.txt
   compute_hessian analytic output_format sparse output hessian.mtx structure_output hessian.xyz

Calculate after a temperature-dependent model has been run at 300 K::

   potential nep.txt
   velocity 300
   ensemble nvt_ber 300 300 100
   time_step 1
   run 1000
   compute_hessian analytic output hessian_300K.out

Caveats
-------

Dense storage scales as :math:`9N^2` doubles on the GPU and host. Dense indices
require :math:`(3N)^2\le2^{31}-1`. Sparse output avoids this dense-matrix limit
but remains subject to ANN, neighbor-list, and sparse-block index bounds and
available memory. Angular workspace is processed in tiles.

The command restores the caller's positions, forces, per-atom energies, and
virials and refreshes force-owned caches at the original geometry, so MD can
continue after the calculation. Temporary descriptor and derivative buffers
are released when the command returns.
