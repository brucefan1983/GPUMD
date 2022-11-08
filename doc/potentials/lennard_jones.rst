.. _lennard_jones_potential:
.. index::
   single: Lennard-Jones potential

Lennard-Jones potential
=======================

The Lennard-Jones (:term:`LJ`) potential is one of the simplest two-body potentials used in :term:`MD` simulations.
The pair potential between particles :math:`i` and :math:`j` is

.. math::
   
   U_{ij} = 4 \epsilon_{ij}
   \left(
   \frac{ \sigma_{ij}^{12} }{ r_{ij}^{12} } -
   \frac{\sigma_{ij}^{6} }{ r_{ij}^{6} }
   \right).

For the implementation in :program:`GPUMD` the potential has neither been shifted nor damped.
This is important to keep in mind when using this model as it implies that both energy and force change discontinuously at the cutoff.

The implementation in :program:`GPUMD` supports up to 10 atom types.

There are two parameters, which respectively control the depth (:math:`\epsilon`) and the position (:math:`\sigma`) of the potential well.
They are given in units of eV and Å, respectively.

File format
-----------

If there is only one atom type, the potential file for this potential model reads::

  lj 1
  epsilon sigma cutoff

Here, :attr:`cutoff` is the cutoff distance.

If there are two atom types, the potential file reads::

  lj 2
  epsilon_00 sigma_00 cutoff_00
  epsilon_01 sigma_01 cutoff_01
  epsilon_10 sigma_10 cutoff_10
  epsilon_11 sigma_11 cutoff_11

If there are three atom types, the potential file reads::
  
  lj 3
  epsilon_00 sigma_00 cutoff_00
  epsilon_01 sigma_01 cutoff_01
  epsilon_02 sigma_02 cutoff_02
  epsilon_10 sigma_10 cutoff_10
  epsilon_11 sigma_11 cutoff_11
  epsilon_12 sigma_12 cutoff_12
  epsilon_20 sigma_20 cutoff_20
  epsilon_21 sigma_21 cutoff_21
  epsilon_22 sigma_22 cutoff_22

The extension to more than three atom types should be apparent from the examples above.
