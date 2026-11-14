.. _kw_import_q_scaler:
.. index::
   single: import_q_scaler (keyword in nep.in)

:attr:`import_q_scaler`
========================

For ordinary structural NEP, when training or resuming without :attr:`fine_tune`, :program:`nep` computes :attr:`q_scaler`, the per-descriptor-component normalization used by the neural network, from the training structures of the current run at generation 0.
This keyword instead tells :program:`nep` to read :attr:`q_scaler` from the :attr:`nep.txt` file present in the run directory, skipping that computation.
This naturally requires such a file to be present.

Syntax
------

::

  import_q_scaler <0 or 1>

Parameters
----------

The flag is 0 (default) or 1. A value of 1 imports the descriptor normalization from ``nep.txt``.

Output
------

None.

Examples
--------

::

  import_q_scaler 1

Notes
-----

Use this option to retain a model's descriptor normalization when continuing training on different data.
Provide ``nep.txt`` with exactly the same architecture and species as the current run.
This option does not extract species or change the search state read from ``nep.restart``.
For extraction from a foundation model, see :ref:`fine_tune <kw_fine_tune>`.

NEP-spin training and restart
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Spin restart automatically restores the scaler from the matching ``nep.txt``,
even with ``import_q_scaler 0``. The model and restart must have compatible
architecture and element order; missing or invalid scalers are errors.

For a fresh Spin search, ``import_q_scaler 1`` imports only the scaler.
Parameters are newly initialized and the energy reference is fitted to the
current training data. On restart, the saved energy reference and search state
are retained. Cross-species ``fine_tune`` remains unsupported for Spin.
See :ref:`nep_spin_scaling` and :ref:`nep_spin_restart` for initialization
and restart details.
