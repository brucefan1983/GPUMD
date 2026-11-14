.. _kw_spin_soc:
.. index::
   single: spin_soc (keyword in nep.in)

:attr:`spin_soc`
==================

Select the rotational symmetry of the spin descriptor::

  spin_soc 1

The default is 1. Value 1 couples spatial and spin tensors into joint-rotation
sectors. Value 0 retains separate spatial and spin ranks and contracts each
full product-tensor norm; the energy is invariant under independent global
rotations of the lattice and spins. Spatial angular information, variable
spin lengths and center/neighbor exchange responses are retained.

Both modes preserve time reversal and have analytical atomic-force,
magnetic-force and virial derivatives, including at zero spin. The flag
specifies the model symmetry; it does not control the DFT reference calculation.
See :ref:`nep_spin`. Each mode defines its own descriptor and parameter layout.
