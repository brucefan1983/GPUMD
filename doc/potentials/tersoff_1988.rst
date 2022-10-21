.. _tersoff_1988:
.. index::
   single: Tersoff potential (1988)

Tersoff potential (1988)
========================

The implementation of the Tersoff (1988) potential in :program:`GPUMD` mimics the one in `lammps <https://lammps.sandia.gov/doc/pair_tersoff.html>`_ [Tersoff1988]_.
The Tersoff-1988 potential has a more general form than the :ref:`Tersoff (1989) potential <tersoff_1989>`.
When possible, it is, however, recommended to use the Tersoff (1989) potential as it faster.

Potential form
--------------

The site potential can be written as

.. math::
   
   U_i =  \frac{1}{2} \sum_{j \neq i} f_C(r_{ij}) \left[ f_R(r_{ij}) - b_{ij} f_A(r_{ij}) \right].

The function :math:`f_{C}` is a cutoff function, which is 1 when :math:`r_{ij}<R` and 0 when :math:`r_{ij}>S` and takes the following form in the intermediate region:

.. math::

   f_{C}(r_{ij}) = \frac{1}{2}
   \left[
   1 + \cos \left( \pi \frac{r_{ij} - R}{S - R} \right)
   \right].

The repulsive function :math:`f_{R}` and the attractive function :math:`f_{A}` take the following forms:

.. math::

   f_{R}(r) &= A e^{-\lambda r_{ij}} \\
   f_{A}(r) &= B e^{-\mu r_{ij}}.

The bond-order is

.. math::

   b_{ij} = \left(1 + \beta^{n} \zeta^{n}_{ij}\right)^{-\frac{1}{2n}},

where

.. math::
   
   \zeta_{ij} &= \sum_{k\neq i, j}f_C(r_{ik}) g_{ijk} e^{\alpha(r_{ij} - r_{ik})^{m}} \\
   g_{ijk} &= \gamma\left( 1 + \frac{c^2}{d^2} - \frac{c^2}{d^2+(h-\cos\theta_{ijk})^2} \right).

.. list-table::
   :header-rows: 1

   * - Parameter
     - Unit
   * - :math:`A`
     - eV
   * - :math:`B`
     - eV
   * - :math:`\lambda`
     - Å\ :math:`^{-1}`
   * - :math:`\mu`
     - Å\ :math:`^{-1}`
   * - :math:`\beta`
     - dimensionless
   * - :math:`n`
     - dimensionless
   * - :math:`c`
     - dimensionless
   * - :math:`d`
     - dimensionless
   * - :math:`h`
     - dimensionless
   * - :math:`R`
     - Å
   * - :math:`S`
     - Å
   * - :math:`m`
     - dimensionless
   * - :math:`\alpha`
     - Å\ :math:`^{-m}`
   * - :math:`\gamma`
     - dimensionless

File format
-----------

We have adopted a file format that similar but not identical to that used by `lammps <https://lammps.sandia.gov/doc/pair_tersoff.html>`_.

The potential file for a single-element system reads::
  
  tersoff_1988 1
  A_000 B_000 lambda_000 mu_000 beta_000 n_000 c_000 d_000 h_000 R_000 S_000 m_000 alpha_000 gamma_000

The potential file for a double-element system reads::
  
  tersoff_1988 2
  A_000 B_000 lambda_000 mu_000 beta_000 n_000 c_000 d_000 h_000 R_000 S_000 m_000 alpha_000 gamma_000
  A_001 B_001 lambda_001 mu_001 beta_001 n_001 c_001 d_001 h_001 R_001 S_001 m_001 alpha_001 gamma_001
  A_010 B_010 lambda_010 mu_010 beta_010 n_010 c_010 d_010 h_010 R_010 S_010 m_010 alpha_010 gamma_010
  A_011 B_011 lambda_011 mu_011 beta_011 n_011 c_011 d_011 h_011 R_011 S_011 m_011 alpha_011 gamma_011
  A_100 B_100 lambda_100 mu_100 beta_100 n_100 c_100 d_100 h_100 R_100 S_100 m_100 alpha_100 gamma_100
  A_101 B_101 lambda_101 mu_101 beta_101 n_101 c_101 d_101 h_101 R_101 S_101 m_101 alpha_101 gamma_101
  A_110 B_110 lambda_110 mu_110 beta_110 n_110 c_110 d_110 h_110 R_110 S_110 m_110 alpha_110 gamma_110
  A_111 B_111 lambda_111 mu_111 beta_111 n_111 c_111 d_111 h_111 R_111 S_111 m_111 alpha_111 gamma_111

The extension to more than two components is accordingly.
