.. _lennard_jones_potential:
.. index::
   single: Lennard-Jones potential

Lennard-Jones potential
=======================

The Lennard-Jones (:term:`LJ`) potential is one of the simplest two-body potentials used in :term:`MD` simulations.

Potential form
--------------

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

The parameter :math:`\epsilon` is the well depth, in eV, and :math:`\sigma` is the zero-crossing distance, in Å.
The untruncated pair potential has its minimum at :math:`r=2^{1/6}\sigma`, where :math:`U=-\epsilon`.

File format
-----------

If there is only one atom type, the potential file for this potential model reads::

  lj 1 element
  epsilon sigma cutoff

Here, :attr:`cutoff` is the cutoff distance and :attr:`element` is the chemical symbol of the element.

If there are two atom types, the potential file reads::

  lj 2 <list of the 2 elements>
  epsilon_00 sigma_00 cutoff_00
  epsilon_01 sigma_01 cutoff_01
  epsilon_10 sigma_10 cutoff_10
  epsilon_11 sigma_11 cutoff_11

For :math:`M` atom types, provide all :math:`M^2` rows, with the second type index changing fastest: ``00, 01, ..., 0(M-1), 10, ..., (M-1)(M-1)``.
GPUMD reads each row explicitly and does not generate cross-species parameters using a mixing rule.
For each pair of species, specify identical parameters for the :math:`ij` and :math:`ji` rows.
