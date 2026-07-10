.. _kw_output_interval:
.. index::
   single: output_interval (keyword in nep.in)

:attr:`output_interval`
========================

This keyword sets the number of generations between writes to :ref:`loss.out <loss_out>` (and the corresponding console output, ``nep.txt``, and test-set output files).

The syntax is::

  output_interval <number_of_generations>

Here, :attr:`<number_of_generations>` must be a positive integer and defaults to :math:`100`.
