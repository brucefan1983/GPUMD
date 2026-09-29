.. _kw_seed:
.. index::
   single: seed (keyword in nep.in)

:attr:`seed`
============

This keyword sets the seed of the random draws of the :term:`SNES` algorithm, which are the initial :math:`\mu` and the population of every generation.
Two runs of one input then produce the same model.
The syntax is::

  seed <seed>

Here, :attr:`<seed>` is a non-negative integer.
Without this keyword the initial :math:`\mu` is seeded from the clock, and two runs of one input differ.
