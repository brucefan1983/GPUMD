.. _kw_spin_l_max:
.. index::
   single: spin_l_max (keyword in nep.in)

:attr:`spin_l_max`
====================

Set the maximum tensor rank.

Syntax
------

::

  spin_l_max L

Parameters
----------

The range is 0--4 and the default is 2. This bounds geometry rank, primitive
spin rank and, in SOC mode, coupled output rank. This setting is independent
of the structural :ref:`l_max <kw_l_max>` and :ref:`spin_order <kw_spin_order>`.

Output
------

None.

Examples
--------

::

  spin_l_max 2

Notes
-----

See :ref:`nep_spin` for the descriptor definition and model contract.
