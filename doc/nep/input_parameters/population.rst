.. _kw_population:
.. index::
   single: population (keyword in nep.in)

:attr:`population`
==================

This keyword sets the size of the population used by the :term:`SNES` algorithm [Schaul2011]_.

Syntax
------

::

  population <population_size>

Parameters
----------

Here, :attr:`<population_size>` sets the population size :math:`N_\mathrm{pop}`, which must satisfy :math:`10 \leq N_\mathrm{pop}\leq 200` and defaults to :math:`N_\mathrm{pop}=50`.

Output
------

None.

Examples
--------

::

  population 50

Notes
-----

If the population size is not divisible by the number of GPUs used for training, it is increased to the next multiple of that number.
This also applies to the default population size.
