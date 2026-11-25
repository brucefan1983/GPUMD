.. _omega2_out:
.. index::
   single: omega2.out (output file)

``omega2.out``
==============

This file contains the squared angular frequencies :math:`\omega^2(\boldsymbol{k})` at different :math:`\boldsymbol{k}`-points.

File format
-----------

The first line begins with ``#`` and lists the cumulative path coordinates of the high-symmetry points, followed by their labels.
Labels for separate paths are separated by ``|``.

The header is followed by ``N_kpoints`` data rows and ``3 * N_basis + 1`` columns, where ``N_basis`` is the number of atoms in ``model.xyz`` before replication, and ``N_kpoints`` is the number of k-points generated from high-symmetry points in the :ref:`kpoints.in input file <kpoints_in>`.

Each data row corresponds to one :math:`\boldsymbol{k}`-point.
Each row contains ``3 * N_basis + 1`` numbers, the first column represents the cumulative distance along the high-symmetry path (in units of Å\ :sup:`-1`), while the remaining columns correspond to the ``3 * N_basis`` values for :math:`\omega^2(\boldsymbol{k})` (in units of THz\ :sup:`2`) in ascending numerical order.

To obtain the ordinary frequency in THz for a nonnegative eigenvalue, use :math:`\nu = \sqrt{\omega^2}/(2\pi)`.
A negative value of :math:`\omega^2` corresponds to an imaginary frequency, with magnitude :math:`\sqrt{-\omega^2}/(2\pi)` in THz.
