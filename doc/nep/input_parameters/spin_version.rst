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
Type-dependent :ref:`spin_cutoff <kw_spin_cutoff>` and :ref:`zbl <kw_zbl>` are
supported. ``fine_tune`` is not supported. ``import_q_scaler`` imports the
scaler while fitting a new energy reference to the current training set.

A complete magnetic descriptor configuration is::

  spin_version     1
  spin_basis_size  8 0
  spin_l_max       2 0 0
  spin_compress    2
  spin_order       3
  spin_soc         1

The last four keywords determine the magnetic descriptor dimension. See
:ref:`nep_spin_dimensions` before changing the shape.
