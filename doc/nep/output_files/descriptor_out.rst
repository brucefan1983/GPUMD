.. _nep_descriptor_out:
.. index::
   single: descriptor.out (output file)

``descriptor.out``
==================

The ``descriptor.out`` file contains the descriptor values for the atoms in the input ``train.xyz`` file.

There are :math:`N` rows and :math:`N_{\rm des}` columns, where :math:`N` is the number of atoms in ``train.xyz`` and :math:`N_{\rm des}` is the dimension for the descriptor vector.

The row index is consistent with the global atom index in the ``train.xyz`` file.

For each row, there are :math:`N_{\rm des}` descriptor components, arranged in a particular order (radial components, three-body angular components, four-body angular components, five-body angular components).
