.. _kw_dump_restart:
.. index::
   single: dump_restart (keyword in run.in)

:attr:`dump_restart`
====================

Write a restart file.

Syntax
------

This keyword only requires a single parameter, which is the output interval (number of steps) of updating the restart file::

  dump_restart <interval>

Example
-------

To update the restart file every 100000 steps for a run, one can add::

  dump_restart 100000

before the :ref:`run keyword <kw_run>`.


Caveats
-------
This keyword is not propagating.
That means, its effect will not be passed from one run to the next.
