.. _kw_seed:
.. index::
   single: seed (keyword in nep.in)

:attr:`seed`
============

Set the random seed for SNES initialization and population sampling.
Using the same seed reproduces these random draws, but does not guarantee identical trained models because GPU summation order can vary.

Syntax
------

::

  seed <seed>

Parameters
----------

Here, :attr:`<seed>` is a non-negative integer.
Without this keyword initialization uses the clock, and the population draws use a fixed seed.

Output
------

None.

Examples
--------

::

  seed 12345

Notes
-----

None.
