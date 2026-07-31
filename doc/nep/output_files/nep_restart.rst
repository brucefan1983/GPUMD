.. _nep_restart:
.. index::
   single: nep.restart (output file)

``nep.restart``
===============

This file enables restarting an optimization run.
If the file is present, training will start from the state saved in this file.
The file is updated during training every :ref:`output_interval <kw_output_interval>` generations.

The file is written every 100 generations so a run shorter than that produces a :ref:`nep.txt <nep_txt>` but no restart file.

The user does not need to understand the contents of this file.
One must, however, ensure that the hyperparameters in the :ref:`nep.in input file <nep_in>` related to the descriptor are the same as those used to generate the restart file.
:program:`nep` checks this: the file holds one row per model parameter and no header, so its row count is compared against the number of parameters that :attr:`nep.in` implies, and a difference is an error reporting both numbers.

The difference between the two numbers is a useful hint as to which hyperparameter differs.
When a :ref:`nep.txt <nep_txt>` is present as well, its header is compared against :attr:`nep.in` keyword by keyword, which identifies the hyperparameter directly; see :ref:`the nep.in page <nep_in>` for what is compared and how mismatches are reported.
Remove this file to start a new training run rather than resuming.
