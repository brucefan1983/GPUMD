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

Spin3 training and restart
--------------------------

Fresh Spin3 training computes scaling from the training structures using a
fixed, non-degenerate reference parameter vector, independent of the SNES
search mean and search RNG. Constant spin projection matrices would collapse
distinct descriptor channels, so the structural NEP constant-vector reference
is not used for Spin3. Response-only structures never enter this calculation.
The :ref:`initial_para <kw_initial_para>` keyword scales the radial coefficients
of this reference model while leaving its channel mixing matrices unchanged.

When ``nep.restart`` is present, Spin3 automatically restores the scaler from
the matching ``nep.txt`` instead of recomputing it.
The two files must come from the same training run with matching architecture
and element order. The scaler must be finite and positive. Missing or
incompatible model data are errors, not requests to initialize a new model.
``import_q_scaler 0`` does not discard the scaler required by a Spin3 restart.

``import_q_scaler 1`` is also supported for a fresh Spin3 search. It imports
scaling only; parameters are still newly initialized and the elemental energy
baseline is fitted to ordinary training data. On restart, both the saved
baseline and search state are retained. The cross-species ``fine_tune`` mode
remains unsupported for Spin3.

``nep.restart`` stores the search distribution and ``energy_reference`` in the
shared restart format (see :ref:`nep_restart`), while ``nep.txt`` stores a
selected candidate model. Spin requires ``model nep_spin``, ``spin_version 1``
and the current shared restart format.
The search mean and that candidate need not predict the same values.
No parameter rescaling or substitution of the selected candidate for the
search mean is performed during restart.

As in the existing SNES lifecycle, a restart begins a new requested generation
segment and resets RNG initialization. The restart file does not store global
generation progress; it is not an exact replay of an uninterrupted trajectory.
