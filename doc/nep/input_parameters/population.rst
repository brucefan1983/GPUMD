.. _kw_population:
.. index::
   single: population (keyword in nep.in)

:attr:`population`
==================

This keyword sets the size of the population used by the :term:`SNES` algorithm [Schaul2011]_.
The syntax is::

  population <population_size>

Here, :attr:`<population_size>` sets the population size :math:`N_\mathrm{pop}`, which must satisfy :math:`10 \leq N_\mathrm{pop}\leq 100` and defaults to :math:`N_\mathrm{pop}=50`.
