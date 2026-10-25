.. _kw_spin_compress:
.. index::
   single: spin_compress (keyword in nep.in)

:attr:`spin_compress`
======================

This keyword sets the shared latent radial rank of the spin descriptor::

  spin_compress <count>

The default is 2 and the input range is 1--9. The symbol :math:`C` denotes
this value in :ref:`nep_spin`. Each channel is a learned combination of the
radial basis, and every Spin3 contraction family reuses the same :math:`C`
channels. It therefore controls descriptor rank and cost; it does not select
a fixed list of Hamiltonian terms.

For ``spin_version 1``, the spin descriptor must not exceed 96 components, and
the combined structural plus spin descriptor must not exceed 199 components.
The exact count also depends on :ref:`spin_order <kw_spin_order>`,
:ref:`spin_l_max <kw_spin_l_max>`, and :ref:`spin_soc <kw_spin_soc>`.

For example, the commonly used order-3, :math:`l_\mathrm{max}=2`, SOC-enabled
shape has 55 magnetic coordinates with::

  spin_compress 2

Together with the default structural descriptors, this default Spin shape has
97 total coordinates and satisfies both dimension limits. Use the formulas in
:ref:`nep_spin_dimensions` before increasing this value.
