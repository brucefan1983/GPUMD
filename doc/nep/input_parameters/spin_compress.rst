.. _kw_spin_compress:
.. index::
   single: spin_compress (keyword in nep.in)

:attr:`spin_compress`
=======================

Set the number of learned radial channels::

  spin_compress C

The default is 3 and the range is 1--9. A learned matrix mixes the structural
angular radial channels into :math:`C` spin channels. All combinations of
``spin_compress 1--9``, ``spin_l_max 0--4``, ``spin_order 1--4`` and
``spin_soc 0/1`` are supported. They require at most 236 magnetic coordinates
(``C=9, L=4, order=4, spin_soc=0``). Together with the existing maximum of
103 structural coordinates, the total limit is 339. The structural angular
limit remains 90; the training input and model reader enforce the same limits.

Models with at most 199 total coordinates keep the smaller ANN workspace;
only larger models use the expanded workspace. The model format and parameter
ordering are unchanged.

For ``spin_l_max 2`` and ``spin_order 2``, SOC gives :math:`1+11C` spin
coordinates; non-SOC gives :math:`1+10C`. See :ref:`nep_spin_dimensions`.
