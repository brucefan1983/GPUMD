.. _kw_batch:
.. index::
   single: batch (keyword in nep.in)

:attr:`batch`
=============

This keyword sets the size of each batch used during the :ref:`optimization procedure <nep_optimization_procedure>`.
The syntax is::

  batch <batch_size>

Here, :attr:`<batch_size>` sets the batch size :math:`N_\mathrm{bat}`, which must satisfy :math:`N_\mathrm{bat}\geq 1` and defaults to :math:`N_\mathrm{bat}=1000`.

In principle one can train against the entire training set during every iteration of the optimization procedure (equivalent to :math:`N_\mathrm{bat}` being identical to the number of structures in the training set).
It is, however, often beneficial for computational speed and potentially necessary for memory reasons to consider only a subset of the training data at any given iteration.
:math:`N_\mathrm{bat}` sets the size of this subset.

Usually, training sets with more diverse structures require using larger batch sizes to achieve maximal accuracy.
In many cases, a batch size between :math:`N_\mathrm{bat}=100` and :math:`N_\mathrm{bat}=1000` should be sufficient to achieve good results.
If you have a powerful GPU (such as a Tesla A100), you can use a large batch size (such as :math:`N_\mathrm{bat}=1000`) or the full-batch (:math:`N_\mathrm{bat}\geq` number of configurations in the training set).
If you use a small batch size for a powerful GPU, you will simply waste the GPU resources.
If you have a weaker GPU, you can use a smaller batch size.

If you observe oscillations in the :term:`RMSE` values for energies, forces, and virials even for generations :math:`\gtrsim 10^5`, it is a sign that the batch size is too small.
