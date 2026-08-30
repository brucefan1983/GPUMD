.. _kw_dump_polarizability:
.. index::
   single: dump_polarizability (keyword in run.in)

:attr:`dump_polarizability`
===========================

Predicts the polarizability for the current configuration of atoms during MD using a separate `nep4_polarizability` model.
The response model is read by this keyword and is independent of the potential used to run the MD.

Syntax
------

.. code::

   dump_polarizability <interval> <nep_file>

:attr:`interval` parameter is the output interval (number of steps) for evaluating and writing the polarizability.

:attr:`nep_file` is the name of the `nep4_polarizability` potential file used to predict the polarizability.

Examples
--------

Example 1
^^^^^^^^^
To use `nep.txt` to propagate the MD and `nep_polarizability.txt` to compute and write the polarizability every 100 steps, write::

  potential nep.txt
  ...
  dump_polarizability 100 nep_polarizability.txt

before the :ref:`run keyword <kw_run>`. This will generate a `polarizability.out` output file containing the time step and predicted polarizability at that time step.


Caveats
-------
* This keyword is not propagating.
  That means, its effect will not be passed from one run to the next.
