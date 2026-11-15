.. _eigenvector_in:
.. index::
   single: gpumd input files; Eigenvectors
   single: eigenvector.in

Modal eigenvectors (``eigenvector.in``)
---------------------------------------

The ``eigenvector.in`` file supplies the normal modes used by :ref:`compute_gkma <kw_compute_gkma>` and :ref:`compute_hnema <kw_compute_hnema>`.
It is a binary file of native 32-bit floating-point values, with no text header or record markers.

File format
^^^^^^^^^^^

For :math:`N` participating atoms, the file contains all :math:`3N` modes in the following order:

1. The :math:`3N` eigenvalues :math:`\omega_n^2`, in ascending order, in units of ps\ :sup:`-2`.
   For a positive eigenvalue, the frequency in THz is :math:`f_n=\sqrt{\omega_n^2}/(2\pi)`.
   Negative eigenvalues correspond to imaginary frequencies.
2. For each mode, the :math:`N` eigenvector components in the :math:`x` direction, followed by the :math:`N` components in the :math:`y` direction and then the :math:`N` components in the :math:`z` direction.

The file therefore contains :math:`3N+9N^2` floating-point values, or :math:`4(3N+9N^2)` bytes.
The eigenvectors are orthonormal eigenvectors of the mass-weighted dynamical matrix; no additional mass factors should be included in the stored vectors.
The atom order and masses must match those used in the modal-analysis simulation.
By default, all atoms participate.
The :attr:`first_mode` and :attr:`last_mode` arguments are one-based, inclusive indices into this full file; selecting a subset of modes does not change the file layout.

Generating the file
^^^^^^^^^^^^^^^^^^^

Use the complete MD simulation cell as the basis for a :ref:`compute_phonon <kw_compute_phonon>` calculation, with the same potential and atom order as the modal-analysis run.
For a ``model.xyz`` file that already contains this complete cell, put ``replicate 1 1 1`` as the first command in ``run.in``.
The ``kpoints.in`` file should contain a single Gamma point::

   0 0 0 G

In this case, ``compute_phonon`` writes ``eigenvector.out`` in the required binary format.
Copy it to the modal-analysis input filename::

   cp eigenvector.out eigenvector.in

If a smaller primitive cell is used as the phonon basis, its eigenvectors cannot be used directly for modal analysis of the larger MD cell.
