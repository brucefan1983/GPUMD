.. _kw_neuron:
.. index::
   single: neuron (keyword in nep.in)

:attr:`neuron`
==============

This keyword sets the number of neurons in the hidden layer of the :term:`NN`.
The syntax is::

  neuron <number_of_neurons>

where :attr:`<number_of_neurons>` corresponds to :math:`N_\mathrm{neu}` in the :ref:`NEP formalism <nep_formalism>` [Fan2022b]_.
Values must satisfy :math:`1 \leq N_\mathrm{neu} \leq 200`.

The default value is 30, which is relatively small but can lead to relatively high speed.
Larger values rarely lead to a notable improvement.
