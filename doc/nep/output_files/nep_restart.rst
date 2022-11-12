.. _nep_restart:
.. index::
   single: nep.restart (output file)

``nep.restart``
===============

This file enables restarting an optimization run.
If the file is present, training will start from the state saved in this file.
The file is continuously updated during training.

The user does not need to understand the contents of this file.
One must, however, ensure that the hyperparameters in the :ref:`nep.in input file <nep_in>` related to the descriptor are the same as those used to generate the restart file.
