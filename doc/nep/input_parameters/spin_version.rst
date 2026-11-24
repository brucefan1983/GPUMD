.. _kw_spin_version:
.. index::
   single: spin_version (keyword in nep.in)

``spin_version``
====================

This keyword specifies the magnetic descriptor version
of the spin-dependent energy model.

Syntax
------

::

  spin_version <version>

Parameters
----------

The only supported value is ``1``, which is also the default for
``model nep_spin``. Omit this keyword for other models.
Version 1 uses the magnetic descriptor described in :ref:`nep_spin`.
The :ref:`spin_order <kw_spin_order>` keyword independently sets the maximum polynomial degree of the spin inputs.

Output
------

None.

Examples
--------

.. _spin_training_example:

The following NEP-spin setup uses the default spin descriptor settings.
All atom types are energy centers; only Fe has independent spin degrees of freedom::

  model            nep_spin
  version          4
  spin_version     1
  type             3 Fe Bi O
  cutoff           6 5
  n_max            4 4
  basis_size       8 8
  l_max            4 2 0
  neuron           30
  spin_type        Fe
  spin_cutoff      6
  spin_compress    3
  spin_l_max       2
  spin_order       2
  spin_soc         1
  lambda_m         1
  batch            512
  generation       100000

See :ref:`nep_spin_dimensions` before changing the descriptor shape.

Notes
-----

Spin training requires ``version 4`` and one hidden layer.
``version`` identifies the NEP architecture; ``spin_version`` identifies the
magnetic descriptor version. The resulting ``nep.txt``
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
