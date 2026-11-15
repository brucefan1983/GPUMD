.. _kw_batch:
.. index::
   single: batch (keyword in nep.in)

:attr:`batch`
=============

This keyword sets the size of each batch used during the :ref:`optimization procedure <nep_optimization_procedure>`.

Syntax
------

::

  batch <batch_size>

Parameters
----------

Here, :attr:`<batch_size>` sets the batch size :math:`N_\mathrm{bat}`, which must satisfy :math:`N_\mathrm{bat}\geq 1` and defaults to :math:`N_\mathrm{bat}=1000`.
This setting does not control :ref:`response.xyz <response_xyz>` data.

Output
------

None.

Examples
--------

::

  batch 1000

Notes
-----

Each generation uses a subset of the training structures. A batch size at least as large as the training set uses the full set.
Smaller batches reduce memory use and work per generation; larger batches can make the loss less noisy and may improve GPU utilization.

Batch sizes of 100--1000 are often a useful starting point. More diverse data may need larger batches.
Check both convergence and memory use for the chosen dataset and GPU.
Persistent oscillations in the training RMSE can indicate that the batch is too small.
