.. _kw_spin_version:
.. index::
   single: spin_version (keyword in nep.in)

:attr:`spin_version`
====================

This keyword selects the spin-dependent energy model and its magnetic
model contract::

  spin_version 1

The only supported value is ``1``. Omit this keyword for ordinary NEP.
A missing value, an unsupported version, or a repeated model selector is an
input error. Version 1 uses the order/compression descriptor described in
:ref:`nep_spin`; :ref:`spin_order <kw_spin_order>` independently selects its
contraction order.

Spin training requires ``version 4``, ``model_type 0``, and one hidden layer.
``version`` identifies the NEP architecture; ``spin_version`` identifies the
magnetic descriptor and parameter-layout contract. The resulting ``nep.txt``
contains ``model nep_spin`` and ``spin_version 1``. Prediction, scaler import,
and restart require a compatible model. The shared model-file keyword
``format_version`` does not belong in ``nep.in``.

Every training frame must contain a Cartesian ``spin:R:3`` property. Add
``mforce:R:3`` to train magnetic forces; see :ref:`train_test_xyz` for aliases
and units. Charge models and type-dependent structural cutoffs are unsupported.
Type-dependent :ref:`spin_cutoff <kw_spin_cutoff>` and universal
:ref:`zbl <kw_zbl>` are supported; flexible ZBL and ``fine_tune`` are unsupported.
See :ref:`import_q_scaler <kw_import_q_scaler>` for scaler import and restart
behavior.

.. _spin_training_example:

Training example
----------------

The following Spin setup uses the default descriptor shape. ``spin_compress``
defaults to 2 and may be omitted. Together with the structural defaults, this
gives 55 magnetic and 97 total coordinates, within :ref:`nep_spin_dimensions`::

  version          4
  type             2 Fe Ge
  spin_version     1
  spin_basis_size  8 0
  spin_l_max       2 0 0
  spin_compress    2
  spin_order       3
  spin_soc         1
  neuron           30
  lambda_m         1.0
  batch            1000
  generation       100000

Each frame in ``train.xyz`` and ``test.xyz`` must then contain a Cartesian
``spin:R:3`` property. Add ``mforce:R:3`` when training magnetic forces. See
:ref:`train_test_xyz` for the complete data contract and :ref:`nep_spin` for
the Spin3 formalism.

See :ref:`nep_spin_dimensions` before changing the descriptor shape.
