.. _kw_save_potential:
.. index::
   single: save_potential (keyword in nep.in)

:attr:`save_potential`
======================

This keyword sets the number of generations between writing a ``nep.txt`` checkpoint file.

Syntax
------

::

  save_potential <number_of_generations_between_save_potential> <format> <save_restart>

Parameters
----------

``number_of_generations_between_save_potential`` sets the checkpoint interval.
``format`` selects the filename format, and ``save_restart`` controls whether to also save a restart checkpoint.

The default number of generations between saved model files is :math:`N=10^5` and the output file names use the extended format.

Output
------

If ``<format>`` is set to ``0`` the output file name is formatted as ``nep_gen[generation].txt``.
If ``<format>`` is set to ``1`` the output file name is formatted as ``nep_y[year]_m[month]_d[day]_h[hour]_m[minute]_s[second]_generation[generation].txt``.
These model files can be used to monitor the training progress of your model.
Additionally, if ``<save_restart>`` is set to ``1`` the :ref:`nep.restart file <nep_restart>` is also written, following the naming format set by ``<format>``.
Note that the :ref:`nep.restart file <nep_restart>` is the file that is required to continue training.

Examples
--------

::

  save_potential 1000 0 1

Notes
-----

None.
