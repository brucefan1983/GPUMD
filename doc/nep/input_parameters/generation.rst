.. _kw_generation:
.. index::
   single: generation (keyword in nep.in)

:attr:`generation`
==================

This keyword sets the number of generations :math:`N_\mathrm{gen}` for the :term:`SNES` algorithm [Schaul2011]_.
The syntax is::

  generation <number_of_generatinons>

Here, :attr:`<number_of_generations>` sets :math:`N_\mathrm{gen}`, which must satisfy :math:`0 \leq N_\mathrm{gen}\leq 10^7` and defaults to :math:`N_\mathrm{gen}=10^5`.
For simple systems, :math:`N_\mathrm{gen}= 10^4 \sim 10^5` is enough.
For more complicated systems, values in the range :math:`N_\mathrm{gen} = 10^5\sim10^6` can be required to obtain a well converged model.
