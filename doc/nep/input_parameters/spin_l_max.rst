.. _kw_spin_l_max:
.. index::
   single: spin_l_max (keyword in nep.in)

:attr:`spin_l_max`
===================

This keyword controls the maximum spatial tensor rank used by the spin
descriptor::

  spin_l_max <l_3body> <l_4body> <l_5body>

The default is ``2 0 0``. For ``spin_version 1``, ``l_3body`` can range from 0
to 2 and the other two values must be zero.

For Spin3, the first value has the following interpretation:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Value
     - Enabled spatial information
   * - 0
     - scalar and spin-only contractions
   * - 1
     - additionally use vector/rank-1 spin--space contractions
   * - 2
     - additionally use symmetric-traceless rank-2 contractions

This rank cutoff is distinct from :ref:`spin_order <kw_spin_order>` and from
the ordinary structural :ref:`l_max <kw_l_max>`. A typical Spin3 setting is::

  spin_l_max 2 0 0
