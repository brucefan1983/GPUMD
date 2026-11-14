.. _kw_output_interval:
.. index::
   single: output_interval (keyword in nep.in)

:attr:`output_interval`
========================

Set the number of generations between updates of the model and training report.

Syntax
------

::

  output_interval <number_of_generations>

Parameters
----------

Here, :attr:`<number_of_generations>` must be a positive integer and defaults to :math:`100`.

Output
------

Updates :ref:`loss.out <loss_out>`, the corresponding console output, :ref:`nep.txt <nep_txt>`, :ref:`nep.restart <nep_restart>`, and test-set prediction files.
The ``*_train.out`` prediction files retain their fixed interval of 1000 generations.

Examples
--------

::

  output_interval 100

Notes
-----

None.
