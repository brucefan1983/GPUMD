.. _omega2_out:
.. index::
   single: omega2.out (output file)

``omega2.out``
==============

This file contains the squared frequencies :math:`\omega^2(\boldsymbol{k})` at different :math:`\boldsymbol{k}`-points.

File format
-----------

There are :attr:`N_kpoints` rows and :attr:`3 * N_basis` columns, where :attr:`N_basis` is the number of basis atoms in the unit cell defined in the :ref:`basis.in input file <basis_in>` and :attr:`N_kpoints` is the number of :math:`\boldsymbol{k}`-points in the :ref:`kpoints.in input file <kpoints_in>`.

Each line corresponds to one :math:`\boldsymbol{k}`-point.
Each line contains :attr:`3 * N_basis` numbers, corresponding to the :attr:`3 * N_basis` values for :math:`\omega^2(\boldsymbol{k})` (in units of THz\ :sup:`2`) in the order of increasing magnitude.
