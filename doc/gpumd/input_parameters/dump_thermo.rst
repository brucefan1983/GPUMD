.. _kw_dump_thermo:
.. index::
   single: dump_thermo (keyword in run.in)

:attr:`dump_thermo`
===================

This keyword controls the writing of global thermodynamic quantities to the :ref:`thermo.out output file <thermo_out>`.

Syntax
------

This keyword only requires a single parameter, which is the output interval (number of steps) of the global thermodynamic quantities::

  dump_thermo <interval>

Example
-------

To dump the global thermodynamic quantities every 1000 steps for a run, one can add::

  dump_thermo 1000

before the :ref:`run keyword <kw_run>`.

Caveats
-------
This keyword is not propagating.
That means, its effect will not be passed from one run to the next.
