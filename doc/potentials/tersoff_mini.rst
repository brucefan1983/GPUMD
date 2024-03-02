.. _tersoff_mini:
.. index::
   single: Tersoff mini-potential

Tersoff mini-potential
======================

The Tersoff-mini potential is described in [Fan2020]_.
It currently only applies to systems with a single atom type.
One can use the `GPUGA potential <https://github.com/brucefan1983/GPUGA>`_ to fit this potential for new systems.

Potential form
--------------

The site potential can be written as

.. math::

   U_i =  \frac{1}{2} \sum_{j \neq i} f_\mathrm{C}(r_{ij}) \left[ f_\mathrm{R}(r_{ij}) - b_{ij} f_\mathrm{A}(r_{ij}) \right].

The function :math:`f_\mathrm{C}` is a cutoff function, which is 1 when :math:`r_{ij}<R_{IJ}` and 0 when :math:`r_{ij}>S_{IJ}` and takes the following form in the intermediate region:

.. math::
   
   f_\mathrm{C}(r_{ij}) = \frac{1}{2}
   \left[
   1 + \cos \left( \pi \frac{r_{ij} - R}{S - R} \right)
   \right].

The repulsive function :math:`f_\mathrm{R}` and the attractive function :math:`f_\mathrm{A}` take the following forms:

.. math::

   f_\mathrm{R}(r_{ij}) &= \frac{D_0}{S-1} \exp\left(\alpha r_0\sqrt{2S} \right) e^{-\alpha\sqrt{2S} r_{ij}} \\
   f_\mathrm{A}(r_{ij}) &= \frac{D_0S}{S-1} \exp\left(\alpha r_0\sqrt{2/S} \right) e^{-\alpha\sqrt{2/S} r_{ij}}.

The bond-order function is

.. math::

   b_{ij} = \left(1 + \zeta^{n}_{ij}\right)^{-\frac{1}{2n}},

where

.. math::

   \zeta_{ij} &= \sum_{k\neq i, j} f_C(r_{ik}) g_{ijk} \\
   g_{ijk} &= \beta \left(h-\cos\theta_{ijk}\right)^2.

Parameters
----------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Unit
   * - :math:`D_0`
     - eV
   * - :math:`\alpha`
     - Å\ :math:`^{-1}`
   * - :math:`r_0`
     - Å
   * - :math:`S`
     - dimensionless
   * - :math:`n`
     - dimensionless
   * - :math:`\beta`
     - dimensionless
   * - :math:`h`
     - dimensionless
   * - :math:`R`
     - Å
   * - :math:`S`
     - Å

File format
-----------

The potential file reads::
  
  tersoff_mini 1 element
  D alpha r0 S beta n h R S

Here, :attr:`element` is the chemical symbol of the element.