.. _kw_plumed:
.. index::
   single: plumed (keyword in run.in)

:attr:`plumed`
===================

Invoke the `PLUMED <https://www.plumed.org/>`_ plugin during a MD run.


Syntax
------

This keyword has the following format::

  plumed <plumed_file> <interval> <if_restart>

The :attr:`plumed_file` parameter is the name of the PLUMED input file.
The :attr:`interval` parameter is the interval (number of steps) of calling the PLUMED subroutines.
The :attr:`if_restart` parameter determines if the PLUMED plugin should restart its calculations, which includes appending to its output files, reading the previous biases, etc.


Requirements and specifications
-------------------------------

* This keyword requires an external package to operate.
  Instructions for how to set up the `PLUMED package <https://www.plumed.org/>`_ can be found :ref:`here <plumed_setup>`.
* Increasing the :attr:`interval` parameter will speed up your simulation if you only want PLUMED to calculate collective variables (CVs). But you have to set it to 1 if your simulation was biased by PLUMED.


Examples
--------

Example 1
^^^^^^^^^

To calculate the distance between atoms, one can add::

  plumed plumed.dat 1000 0

before the :ref:`run command <kw_run>`. The plumed.dat file, for example, may look like::

  UNITS LENGTH=A TIME=fs ENERGY=eV
  FLUSH STRIDE=1
  DISTANCE ATOMS=1,2 LABEL=d1
  PRINT FILE=colvar ARG=d1

The output file created by PLUMED (the colvar file) will be updated every 1000 steps.

Example 2
^^^^^^^^^

To restrain the distance between atoms, one can add::

  plumed plumed.dat 1 0

before the :ref:`run command <kw_run>`. The plumed.dat file, for example, may look like::

  UNITS LENGTH=A TIME=fs ENERGY=eV
  FLUSH STRIDE=1
  DISTANCE ATOMS=1,2 LABEL=d1
  RESTRAINT ARG=d1 AT=0.0 KAPPA=2.0 LABEL=restraint
  PRINT FILE=colvar ARG=d1,restraint.bias


Caveats
-------

* This keyword is not propagating.
  That means, its effect will not be passed from one run to the next.
