.. _kw_compute_hessian:
.. index::
   single: compute_hessian (keyword in run.in)

:attr:`compute_hessian`
=======================

Compute the Cartesian Hessian analytically for the current atomic positions
using a single NEP energy model. The command executes as soon as it is read
and must follow :ref:`potential <kw_potential>`. It does not relax the
structure. For ordinary NEP, no preceding MD run is needed.

Syntax
------

.. code-block:: text

   compute_hessian analytic [<option> <value> ...]

Write the entire command on one line in ``run.in``; backslash continuation
is not supported. Options are pairs of names and values as listed below.

Parameters
----------

``analytic``
   Required calculation method.

``output <path>``
   Matrix output path. Default: ``hessian.out``.

``output_format dense|sparse``
   Matrix format. Default: ``dense`` (full text matrix). ``sparse`` writes
   Matrix Market coordinate format.

``raw_output <path>``
   Optional matrix before symmetrization, in the selected output format.
   Default: no raw file.

``metadata <path>``
   Optional JSON file containing system information, energy, forces, matrix
   diagnostics, storage format, and model temperature when applicable.
   Default: no metadata file. Available with both matrix formats.

``structure_output <path>``
   Optional extended XYZ snapshot containing positions, species, masses,
   and forces. Default: no snapshot. Available with both matrix formats.

Each execution overwrites the selected output files. Choose distinct paths
for all outputs; they must also differ from ``model.xyz``, ``run.in``, and
potential files. A file-write error terminates the calculation.

.. _hessian_out:

Output
------

Matrix definition
^^^^^^^^^^^^^^^^^

For :math:`N` atoms, the Hessian is a :math:`3N\times3N` matrix:

.. math::

   H_{i\alpha,j\beta}
   = -\frac{\partial F_{i\alpha}}{\partial R_{j\beta}}
   = \frac{\partial^2 E}{\partial R_{i\alpha}\partial R_{j\beta}}.

Its units are :math:`\mathrm{eV}/\text{\AA}^2`. Rows and columns use the
coordinate order

.. math::

   x_0,\ldots,x_{N-1},y_0,\ldots,y_{N-1},z_0,\ldots,z_{N-1}.

Atom indices follow the input structure order. The main output is
:math:`(H+H^\mathsf{T})/2`; ``raw_output`` contains the matrix before this
symmetrization. Neither output is mass weighted, and no acoustic sum rule
is imposed. For mode analysis, apply mass weighting separately as
:math:`D_{i\alpha,j\beta}=H_{i\alpha,j\beta}/\sqrt{m_i m_j}`.
The command does not calculate phonon frequencies or eigenvectors.

For periodic systems, contributions from different periodic images of the
same atom are already summed. Neither dense nor sparse output retains
lattice-translation labels. A matrix from a small periodic box therefore
cannot be used directly as translation-resolved real-space force constants
for a phonon dispersion at arbitrary wave vectors. For phonon dispersions,
see :ref:`compute_phonon <kw_compute_phonon>` and its cell-size requirements.

Dense format
^^^^^^^^^^^^

The default ``hessian.out`` begins with ``#`` comment lines describing the
units, ordering, atom count, and result type, followed by one matrix row per
line. Read it with NumPy::

   import numpy as np
   H = np.loadtxt("hessian.out")

.. _hessian_sparse:

Sparse format
^^^^^^^^^^^^^

The Matrix Market file begins with::

   %%MatrixMarket matrix coordinate real general

After ``%`` comment lines, a dimensions line gives ``3N 3N nnz``. The
remaining lines contain ``row column value`` with one-based indices in the
coordinate order above. Both halves of the symmetric matrix are stored;
exactly zero entries are omitted. Read it without converting to dense::

   from scipy.io import mmread
   H = mmread("hessian.mtx").tocsr()

Diagnostics
^^^^^^^^^^^

Before symmetrization, the command checks that the matrix is finite,
symmetric within numerical tolerance, and consistent with uniform
translations producing no restoring force. These checks help detect
unreliable results; they do not establish that the structure is relaxed
or dynamically stable.

Failed checks terminate the calculation. If the matrix is finite but fails
symmetry or translation checks, requested ``raw_output`` and ``metadata``
files are still written and marked ``failed_validation``. The normal
matrix and structure snapshot are not written in this case. Metadata
records the failure reason and raw diagnostics; symmetrized diagnostics
are ``null``. Do not treat these diagnostic files as an accepted result.

Examples
--------

Write the default dense ``hessian.out`` for the positions in ``model.xyz``::

   potential nep.txt
   compute_hessian analytic

Write a sparse matrix and a structure snapshot::

   potential nep.txt
   compute_hessian analytic output_format sparse output hessian.mtx structure_output hessian.xyz

Also request the raw matrix and diagnostics (valid with either format)::

   potential nep.txt
   compute_hessian analytic raw_output hessian_raw.out metadata hessian.json

For a temperature-dependent NEP model, run at a constant target temperature
before evaluating the final current configuration::

   potential nep.txt
   velocity 300
   ensemble nvt_ber 300 300 100
   time_step 1
   run 1000
   compute_hessian analytic output hessian_300K.out metadata hessian_300K.json

.. _hessian_analytic:

Notes
-----

* Supported models are ordinary and temperature-dependent NEP energy models
  on a single GPU, including their ZBL terms. Multiple potentials, multi-GPU
  NEP, charge, spin, vdW, and ILP models are unsupported. DFT-D3 is rejected
  wherever its command appears in ``run.in``. This command has no
  finite-difference mode or fallback.
* For temperature-dependent NEP, the model temperature is held fixed at
  the value reached by the preceding ``run``, as determined by its ensemble
  target temperatures. Before any ``run`` it is zero; ``velocity`` alone
  does not set it. The actual value is reported in the log and metadata.
  The result describes the current configuration at this fixed model
  temperature, without thermal averaging or phonon renormalization.
* Dense storage requires :math:`9N^2` matrix entries. Choose sparse output
  for large systems when most entries are zero; its storage grows with the
  number of interacting atom pairs. Both formats remain limited by
  available memory and supported system sizes.
