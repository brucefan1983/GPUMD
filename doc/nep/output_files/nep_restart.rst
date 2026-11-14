.. _nep_restart:
.. index::
   single: nep.restart (output file)

``nep.restart``
===============

This file enables restarting an optimization run.
If the file is present, training will start from the state saved in this file.
The file is updated during training every :ref:`output_interval <kw_output_interval>` generations.

The default :ref:`output_interval <kw_output_interval>` is 100 generations.

The file uses the `NEP restart format
<https://github.com/brucefan1983/GPUMD/blob/master/potentials/nep/nep_model_format.md#6-neprestart>`_.

Remove this file to start a new training run rather than resuming.

For spin models, keep the matching :ref:`nep.txt <nep_txt>` alongside ``nep.restart``.
