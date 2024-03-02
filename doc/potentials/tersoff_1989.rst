.. _tersoff_1989:
.. index::
   single: Tersoff potential (1989)

Tersoff potential (1989)
========================

The Tersoff (1989) potential supports systems with one or two atom types [Tersoff1989]_.
It is less general than the :ref:`Tersoff (1988) potential <tersoff_1988>` but faster.


Potential form
--------------

Below we use :math:`i,j,k,\cdots` for atom indices and :math:`I,J,K,\cdots` for atom types.

The site potential can be written as

.. math::

   U_i =  \frac{1}{2} \sum_{j \neq i} f_\mathrm{C}(r_{ij}) \left[ f_\mathrm{R}(r_{ij}) - b_{ij} f_\mathrm{A}(r_{ij}) \right].

The function :math:`f_\mathrm{C}` is a cutoff function, which is 1 when :math:`r_{ij}<R_{IJ}` and 0 when :math:`r_{ij}>S_{IJ}` and takes the following form in the intermediate region:

.. math::

   f_\mathrm{C}(r_{ij}) = \frac{1}{2}
   \left[
   1 + \cos \left( \pi \frac{r_{ij} - R_{IJ}}{S_{IJ} - R_{IJ}} \right)
   \right].

The repulsive function :math:`f_\mathrm{R}` and the attractive function :math:`f_\mathrm{A}` take the following forms:

.. math::

   f_\mathrm{R}(r) &= A_{IJ} e^{-\lambda_{IJ} r_{ij}} \\
   f_\mathrm{A}(r) &= B_{IJ} e^{-\mu_{IJ} r_{ij}}.

The bond-order function is

.. math::

   b_{ij} = \chi_{IJ} \left(1 + \beta_{I}^{n_{I}} \zeta^{n_{I}}_{ij}\right)^{-\frac{1}{2n_{I}}},

where

.. math::

   \zeta_{ij} &= \sum_{k\neq i, j} f_\mathrm{C}(r_{ik}) g_{ijk} \\
   g_{ijk} &= 1 + \frac{c_{I}^2}{d_{I}^2} - \frac{c_{I}^2}{d_{I}^2+(h_{I}-\cos\theta_{ijk})^2}.



.. list-table::
   :header-rows: 1

   * - Parameter
     - Unit
   * - :math:`A_{IJ}`
     - eV
   * - :math:`B_{IJ}`
     - eV
   * - :math:`\lambda_{IJ}`
     - Å\ :math:`^{-1}`
   * - :math:`\mu_{IJ}`
     - Å\ :math:`^{-1}`
   * - :math:`\beta_I`
     - dimensionless
   * - :math:`n_I`
     - dimensionless
   * - :math:`c_I`
     - dimensionless
   * - :math:`d_I`
     - dimensionless
   * - :math:`h_I`
     - dimensionless
   * - :math:`R_{IJ}`
     - Å
   * - :math:`S_{IJ}`
     - Å
   * - :math:`\chi_{IJ}`
     - dimensionless

File format
-----------

Single-element systems
^^^^^^^^^^^^^^^^^^^^^^

In this case, :math:`\chi_{IJ}` is irrelevant. The potential file reads::
  
  tersoff_1989 1 <element>
  A B lambda mu beta n c d h R S
  
Here, :attr:`element` is the chemical symbol of the element.

Two-element systems
^^^^^^^^^^^^^^^^^^^

In this case, there are two sets of parameters, one for each atom type.
The following mixing rules are used to determine some parameters between the two atom types :math:`i` and :math:`j`:

.. math::

   A_{IJ} &= \sqrt{A_{II} A_{JJ}} \\
   B_{IJ} &= \sqrt{B_{II} B_{JJ}} \\
   R_{IJ} &=  \sqrt{R_{II} R_{JJ}} \\
   S_{IJ} &=  \sqrt{S_{II} S_{JJ}} \\
   \lambda_{IJ} &=  (\lambda_{II} + \lambda_{JJ})/2 \\
   \mu_{IJ} &= (\mu_{II} + \mu_{JJ})/2.

Here, the parameter :math:`\chi_{01}=\chi_{10}` needs to be provided.
:math:`\chi_{00}=\chi_{11}=1` by definition.

The potential file reads::
  
  tersoff_1989 2 <list of the 2 elements>
  A_0 B_0 lambda_0 mu_0 beta_0 n_0 c_0 d_0 h_0 R_0 S_0
  A_1 B_1 lambda_1 mu_1 beta_1 n_1 c_1 d_1 h_1 R_1 S_1
  chi_01
