.. _kw_dump_dipole:
.. index::
   single: dump_dipole (keyword in run.in)

:attr:`dump_dipole`
=====================

Predicts the dipole for the current configuration of atoms during MD, using the second supplied NEP.
The first potential should be a regular NEP potential model and is used to run the MD, whilst the second NEP should be a `nep*_dipole` model.

Syntax
------

.. code::

   dump_dipole  <interval>

:attr:`interval` parameter is the output interval (number of steps) for evaluating and writing the dipole.

Examples
--------

Example 1
^^^^^^^^^
To use one NEP potential to propagate the MD (`nep0`), and another (`nep1`) to compute and write the dipole every 100 steps, write::

  potential nep0
  potential nep1  
  ...
  dump_dipole 100

before the :ref:`run keyword <kw_run>`. This will generate a `dipole.out` output files containing the time step and predicted dipole at that time step.


Caveats
-------
* This keyword is not propagating.
  That means, its effect will not be passed from one run to the next.
