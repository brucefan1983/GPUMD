.. _kw_import_q_scaler:
.. index::
   single: import_q_scaler (keyword in nep.in)

:attr:`import_q_scaler`
========================

For ordinary structural NEP, when training or resuming without :attr:`fine_tune`, :program:`nep` computes :attr:`q_scaler`, the per-descriptor-component normalization used by the neural network, from the training structures of the current run at generation 0.
This keyword instead tells :program:`nep` to read :attr:`q_scaler` from the :attr:`nep.txt` file present in the run directory, skipping that computation.
This naturally requires such a file to be present.
The syntax is::

  import_q_scaler <0 or 1>

with a default value of 0.

This is intended for resuming training from a restart state (:attr:`nep.restart`) that was transplanted from a different model, e.g. a foundation model restricted to a subset of species via a species-selection tool.
In that case, the transplanted network weights were tuned under the :attr:`q_scaler` of the original model, and recomputing a different one from a smaller or less diverse training set can force the weights to spend part of the optimization readapting to the new descriptor scale, rather than only to the new training data.
Unlike :attr:`fine_tune`, this keyword does not perform any species extraction: the :attr:`nep.txt` file it reads from is expected to already have the exact same architecture and species as the current run, which is why the file is referred to implicitly rather than by a path argument, matching the way :attr:`nep.restart` is referred to.
:program:`nep` compares the header of that file against :attr:`nep.in` and raises an error naming the keyword on any mismatch; see :ref:`the nep.in page <nep_in>` for the details.
The mean/standard-deviation restart state (:attr:`mu`/:attr:`sigma`) is unaffected by this keyword: it is unconditionally read from :attr:`nep.restart`, exactly as in any other non-:attr:`fine_tune` run.

Spin training and restart
-------------------------

Spin restart automatically restores the scaler from the matching ``nep.txt``,
even with ``import_q_scaler 0``. The model and restart must have compatible
architecture and element order; missing or invalid scalers are errors.

For a fresh Spin search, ``import_q_scaler 1`` imports only the scaler.
Parameters are newly initialized and the energy reference is fitted to the
current training data. On restart, the saved energy reference and search state
are retained. Cross-species ``fine_tune`` remains unsupported for Spin.
See :ref:`nep_spin_scaling` and :ref:`nep_spin_restart` for initialization
and restart details.
