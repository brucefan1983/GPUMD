.. _kw_checkpoint:
.. index::
   single: checkpoint (keyword in nep.in)

:attr:`checkpoint`
==================

This keyword sets the number of of generations between writing a checkpoint ``nep.txt`` file, with the name formatted as ``nep_y[year]_m[month]_d[day]_h[hour]_m[minute]_s[second]_generation[generation].txt``.
Checkpoint model files can be used to monitor the training progress of your model.

Note that the :ref:`nep.restart file <nep_restart>` is the file that is required to continue training.
The syntax is::

  checkpoint <number_of_generations_between_checkpoints>

The default number of generations between checkpoint files is :math:`N=10^5`.
