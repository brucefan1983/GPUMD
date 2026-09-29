.. _kw_seed:
.. index::
   single: seed (keyword in nep.in)

:attr:`seed`
============

This keyword sets the seed of the random draws of the :term:`SNES` algorithm, which are the initial :math:`\mu` and the population of every generation.
With this keyword the forces and the Born effective charges are also summed in fixed point, and their sums do not depend on the order in which the GPU adds the contributions.
Two runs of one input on the same GPU then produce the same model.
The syntax is::

  seed <seed>

Here, :attr:`<seed>` is a non-negative integer.
Without this keyword the initial :math:`\mu` is seeded from the clock, the population draws use a fixed seed, and the forces and Born effective charges are summed in floating point.
