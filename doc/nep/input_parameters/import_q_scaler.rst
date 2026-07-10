.. _kw_import_q_scaler:
.. index::
   single: import_q_scaler (keyword in nep.in)

:attr:`import_q_scaler`
========================

By default, when training or resuming a NEP potential without :attr:`fine_tune`, :program:`nep` computes :attr:`q_scaler`, the per-descriptor-component normalization used by the neural network, from the training structures of the current run at generation 0.
This keyword instead tells :program:`nep` to read :attr:`q_scaler` from the :attr:`nep.txt` file present in the run directory, skipping that computation.
This naturally requires such a file to be present.
The syntax is::

  import_q_scaler <0 or 1>

with a default value of 0.

This is intended for resuming training from a restart state (:attr:`nep.restart`) that was transplanted from a different model, e.g. a foundation model restricted to a subset of species via a species-selection tool.
In that case, the transplanted network weights were tuned under the :attr:`q_scaler` of the original model, and recomputing a different one from a smaller or less diverse training set can force the weights to spend part of the optimization readapting to the new descriptor scale, rather than only to the new training data.
Unlike :attr:`fine_tune`, this keyword does not perform any species extraction: the :attr:`nep.txt` file it reads from is expected to already have the exact same architecture and species as the current run (:program:`nep` validates this and raises an error on a mismatch), which is why the file is referred to implicitly rather than by a path argument, matching the way :attr:`nep.restart` is referred to.
The mean/standard-deviation restart state (:attr:`mu`/:attr:`sigma`) is unaffected by this keyword: it is unconditionally read from :attr:`nep.restart`, exactly as in any other non-:attr:`fine_tune` run.
