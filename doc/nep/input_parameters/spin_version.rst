.. _kw_spin_version:
.. index::
   single: spin_version (keyword in nep.in)

:attr:`spin_version`
======================

Select the spin descriptor and parameter layout.

Syntax
------

::

  model nep_spin
  spin_version 1

Parameters
----------

The supported value is 1, which is also the default for ``model nep_spin``.
Spin training uses NEP ``version 4`` and one hidden layer. ``format_version``
identifies the model-file syntax and is not an input keyword in ``nep.in``.

Every training frame must contain Cartesian ``spin:R:3``. Add ``mforce:R:3``
to train magnetic forces; see :ref:`train_test_xyz` for aliases and units.
Type-dependent :ref:`spin_cutoff <kw_spin_cutoff>` and universal
:ref:`zbl <kw_zbl>` are supported. Charge models, type-dependent structural
cutoffs, flexible ZBL and ``fine_tune`` are unsupported.

Output
------

The exported ``nep.txt`` records the spin descriptor settings.

.. _spin_training_example:

Examples
--------

This setup has 30 structural and 34 spin coordinates (64 total), with 6874
parameters. All types are energy centers; only Fe has independent spins::

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

Defaults are ``spin_compress 3``, ``spin_l_max 2``, ``spin_order 2`` and
``spin_soc 1``. In this example, ``basis_size 8 8`` gives the spin
descriptor nine radial basis functions.
For ``spin_soc 0``, this example has 61 total coordinates and 6568 parameters.

Notes
-----

See :ref:`nep_spin` for the model definition and restart requirements.
