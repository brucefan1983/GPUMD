.. _kw_model:
.. index::
   single: model (keyword in nep.in)

:attr:`model`
=============

This keyword selects the model used for training or prediction::

  model <name>

The names match the ``model`` field in ``nep.txt``. If no model-selection
keyword is given, the default is ``nep``.

.. list-table::
   :header-rows: 1

   * - Name
     - Model
     - Equivalent legacy selection
   * - ``nep``
     - Standard energy NEP
     - ``model_type 0`` (or no selector)
   * - ``nep_dipole``
     - TNEP dipole
     - ``model_type 1``
   * - ``nep_polarizability``
     - TNEP polarizability
     - ``model_type 2``
   * - ``nep_temperature``
     - Temperature-dependent NEP
     - ``model_type 3``
   * - ``nep_spin``
     - Spin-dependent energy NEP
     - ``spin_version 1``
   * - ``qnep1``
     - qNEP mode 1
     - ``charge_mode 1``
   * - ``qnep2``
     - qNEP mode 2
     - ``charge_mode 2``
   * - ``qnep3``
     - qNEP mode 2 with charge self-energy correction
   * - ``nep_vdw``
     - NEP with learned long-range vdW
     - ``vdw 1``
   * - ``qnep3_vdw``
     - qNEP with charge self-energy excluded and learned long-range vdW
     - ``charge_vdw 1``

The new keyword selects the same implementations as the legacy inputs;
in particular, the definitions of ``qnep1`` and ``qnep2`` are unchanged.
``qnep3`` uses the same parameter layout as ``qnep2``, but adds the
self-energy correction to the long-range charge energy.

Specify ``model`` at most once. It cannot be combined with ``model_type``,
``charge_mode``, ``vdw``, or ``charge_vdw``, even if a legacy value is zero.
This rule is independent of keyword order.

For ``model nep_spin``, :ref:`spin_version <kw_spin_version>` may still be
specified and defaults to ``1``. It is an error to combine ``spin_version``
with any other explicit ``model``.

Legacy model-selection keywords remain supported and print a deprecation
warning with the equivalent ``model`` selection. Using ``spin_version``
without ``model`` also remains supported with a warning; ``spin_version``
itself remains the spin descriptor version setting.

To migrate ``charge_mode 2 1``, use::

  model qnep2
  flip_charge 1

See :ref:`flip_charge <kw_flip_charge>` for the restart option.
