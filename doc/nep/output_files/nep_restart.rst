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

Spin3 restarts use this shared format and preserve ``energy_reference`` together
with the search mean and standard deviation. A matching ``nep.txt`` is still
needed for the Spin3 metadata and descriptor scalers. Spin models must use
``model nep_spin`` and ``spin_version 1``; only the current shared restart
format is accepted.
