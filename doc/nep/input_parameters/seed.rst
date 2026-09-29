.. _kw_seed:
.. index::
   single: seed (keyword in nep.in)

:attr:`seed`
============

This keyword sets the seed of the random draws of the :term:`SNES` algorithm, which are the initial :math:`\mu` and the population of every generation.
Two runs of one input then draw the same random numbers.
The forces are summed on the GPU in an order that varies between runs.
Two runs of one input can therefore produce different models after many generations.
The syntax is::

  seed <seed>

Here, :attr:`<seed>` is a non-negative integer.
Without this keyword the initial :math:`\mu` is seeded from the clock, and the population draws use a fixed seed.
