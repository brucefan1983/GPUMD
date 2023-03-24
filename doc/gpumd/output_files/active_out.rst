.. _active_out:
.. index::
   single: active.out (output file)

``active.out``
==============

This file contains the simulation time :math:`t` and uncertainty :math:`\sigma_f` for each step of the MD simulation that has been checked during active learning.
This file is generated when the :ref:`active keyword <kw_active>` is invoked, which also controls the frequency with which to check the uncertainty.

Note that the time and uncertainty will be written to this file regardless of if the structure exceeds the threshold :math:`\delta`.


File format
-----------

There are two columns in this file::

  column   1 2
  quantity t :math:`\sigma_f`

where the first column is the time in fs, and the second is the observed uncertainty in eV/Å.


