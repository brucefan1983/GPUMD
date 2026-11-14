.. _descriptor_out:
.. index::
   single: descriptor.out (output file)

``descriptor.out``
==================

The ``descriptor.out`` file contains the per-structure or per-atom descriptor values for the input ``train.xyz`` file.

This file is written only in :ref:`prediction mode <kw_prediction>` (``prediction 1``), with :ref:`output_descriptor <kw_output_descriptor>` set to 1 or 2.
It is not written during training.
Each prediction run appends to an existing file; remove or rename it before running if only the new results are wanted.
Mode 1 writes one row per structure; mode 2 writes one row per atom.

File format
-----------

Each prediction run writes :math:`N` rows and :math:`N_{\rm des}` columns, where :math:`N` is the number of structures or atoms in ``train.xyz`` and :math:`N_{\rm des}` is the dimension for the descriptor vector.

Within each run's output, the row index is consistent with the global structure or atom index in the ``train.xyz`` file.

Each row contains the normalized descriptor components used by the model.
Mode 1 averages these components over the atoms in each structure.

For ordinary NEP, radial components come first, followed by three-body angular components.
The remaining invariant blocks follow the switch order in :ref:`l_max <kw_l_max>`:
``q222``, ``q1111``, ``q112``, ``q123``, ``q233``, and ``q134``, omitting disabled blocks.
Within each angular block, the radial index increases from 0 to :math:`n_\mathrm{max}^\mathrm{A}`.
For NEP-spin, the magnetic descriptor components follow the structural components.
