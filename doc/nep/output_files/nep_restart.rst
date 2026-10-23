.. _nep_restart:
.. index::
   single: nep.restart (output file)

``nep.restart``
===============

This file enables restarting an optimization run.
If the file is present, training will start from the state saved in this file.
The file is updated during training every :ref:`output_interval <kw_output_interval>` generations.

The default :ref:`output_interval <kw_output_interval>` is 100 generations.

The user does not need to understand the contents of this file.

Remove this file to start a new training run rather than resuming.
