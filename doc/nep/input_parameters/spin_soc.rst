.. _kw_spin_soc:
.. index::
   single: spin_soc (keyword in nep.in)

:attr:`spin_soc`
=================

This Spin3 keyword controls whether the descriptor can couple spin direction
to the local spatial environment::

  spin_soc <flag>

The flag can be 0 or 1 and defaults to 1. It is used only with
``spin_version 1``.

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Value
     - Rotational symmetry represented by the descriptor
   * - 0
     - independent rotations of positions and spins
   * - 1
     - joint rotations of positions and axial spin vectors (default)

With value 0, Spin3 retains only the separately spin-rotation-invariant
subset. With value 1, it also includes scalar contractions between spin and
spatial rank-1/rank-2 objects up to
:ref:`spin_l_max <kw_spin_l_max>`. The flag changes the descriptor's allowed
information flow; it does not impose a particular analytic SOC Hamiltonian.
