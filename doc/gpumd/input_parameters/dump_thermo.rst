.. _kw_dump_thermo:
.. index::
   single: dump_thermo (keyword in run.in)

:attr:`dump_thermo`
===================

This keyword controls the writing of global thermodynamic quantities to the :ref:`thermo.out output file <thermo_out>`.

Syntax
------

::

  dump_thermo <interval>

Parameters
----------

``interval`` is the number of MD steps between outputs.

Output
------

Global thermodynamic quantities are written to :ref:`thermo.out <thermo_out>`.

Examples
--------

To dump the global thermodynamic quantities every 1000 steps for a run, one can add::

  dump_thermo 1000

before the :ref:`run keyword <kw_run>`.

Notes
-----

Repeat this command before each run that needs this output.
