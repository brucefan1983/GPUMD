.. _kw_dump_dipole:
.. index::
   single: dump_dipole (keyword in run.in)

:attr:`dump_dipole`
===================

Predicts the dipole for the current configuration of atoms during MD using a separate `nep4_dipole` model.
The response model is read by this keyword and is independent of the potential used to run the MD.

Syntax
------

.. code::

   dump_dipole <interval> <nep_file>

:attr:`interval` parameter is the output interval (number of steps) for evaluating and writing the dipole.

:attr:`nep_file` is the name of the `nep4_dipole` potential file used to predict the dipole.

Examples
--------

Example 1
^^^^^^^^^
To use `nep.txt` to propagate the MD and `nep_dipole.txt` to compute and write the dipole every 100 steps, write::

  potential nep.txt
  ...
  dump_dipole 100 nep_dipole.txt

before the :ref:`run keyword <kw_run>`. This will generate a `dipole.out` output file containing the time step and predicted dipole at that time step.


Caveats
-------
* This keyword is not propagating.
  That means, its effect will not be passed from one run to the next.
