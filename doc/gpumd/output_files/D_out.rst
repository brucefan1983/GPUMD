.. _D_out:
.. index::
   single: D.out (output file)

``D.out``
=========


This file contains the dynamical matrices :math:`D(\boldsymbol{k})` at different :math:`\boldsymbol{k}`-points.

File format
-----------

The file contains :attr:`3 * N_basis * N_kpoints` rows and :attr:`6 * N_basis` columns, where :attr:`N_basis` is the number of basis atoms in the unit cell defined in the :ref:`basis.in input file <basis_in>` and :attr:`N_kpoints` is the number of :math:`\boldsymbol{k}`-points in the :ref:`kpoints.in input file <kpoints_in>`.

Each consecutive :attr:`3 * N_basis` rows correspond to one :math:`\boldsymbol{k}`-point.
For each :math:`\boldsymbol{k}`-point, the first :attr:`3*N_basis` columns correspond to the real part and the second :attr:`3*N_basis` columns correspond to the imaginary part.
Schematically, the file is organized as::

  D_real(k_0) D_imag(k_0)
  D_real(k_1) D_imag(k_1)
  ...
 
The matrix elements are in units of eV Å\ :math:`^{-2}` amu\ :math:`^{-1}`.
