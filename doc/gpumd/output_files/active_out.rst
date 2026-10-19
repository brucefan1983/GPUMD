.. _active_out:
.. index::
   single: active.out (output file)

``active.out``
==============

This file contains the simulation time :math:`t` and uncertainty :math:`\sigma_f` for each step of the MD simulation that has been checked during active learning.
This file is generated when the :ref:`active keyword <kw_active>` is invoked, which also controls the frequency with which to check the uncertainty.

The time and uncertainty are written to this file whether or not the uncertainty exceeds the threshold :math:`\delta`.
The uncertainty is NaN when a model gives forces that are not finite.


File format
-----------

There are two columns in this file::

  column   1 2
  quantity t s

where the first column is the time in fs, and the second is the observed uncertainty in eV/Å.


