.. _kw_flip_charge:
.. index::
   single: flip_charge (keyword in nep.in)

:attr:`flip_charge`
===================

This keyword controls charge-sign reversal when restarting training::

  flip_charge <flag>

The flag is ``0`` (default) or ``1``. When set to ``1``, the charge-output
weights in the search mean loaded from ``nep.restart`` are negated. It has
no effect on fresh initialization or prediction.

This option is supported for ``qnep1``, ``qnep2``, ``qnep3``, and ``qnep3_vdw``. For example::

  model qnep2
  flip_charge 1

The optional second parameter of the legacy :ref:`charge_mode <kw_charge_mode>`
keyword retains the same meaning. Do not specify both that parameter and
``flip_charge``, or repeat the ``flip_charge`` keyword.
