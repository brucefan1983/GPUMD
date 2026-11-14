.. _descriptor_out:
.. index::
   single: descriptor.out (output file)

``descriptor.out``
==================

The ``descriptor.out`` file contains the per-structure or per-atom descriptor values for the input ``train.xyz`` file.

The file is generated in :ref:`prediction mode <kw_prediction>` with :ref:`output_descriptor <kw_output_descriptor>`.
Mode 1 writes one row per structure; mode 2 writes one row per atom.

File format
-----------

There are :math:`N` rows and :math:`N_{\rm des}` columns, where :math:`N` is the number of structures or atoms in ``train.xyz`` and :math:`N_{\rm des}` is the dimension for the descriptor vector.

The row index is consistent with the global structure or atom index in the ``train.xyz`` file.

For each row, there are :math:`N_{\rm des}` descriptor components, arranged in a particular order (radial components, three-body angular components, four-body angular components, five-body angular components).
