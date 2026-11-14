.. _kw_spin_compress:
.. index::
   single: spin_compress (keyword in nep.in)

:attr:`spin_compress`
=======================

Set the number of learned radial channels::

  spin_compress C

The default is 3 and the range is 1--9. A learned matrix mixes the structural
angular radial channels into :math:`C` spin channels. The descriptor is limited
to 96 spin and 199 total coordinates.

For ``spin_l_max 2`` and ``spin_order 2``, SOC gives :math:`1+11C` spin
coordinates; non-SOC gives :math:`1+10C`. See :ref:`nep_spin_dimensions`.
