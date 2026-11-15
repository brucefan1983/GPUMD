.. _kw_spin_compress:
.. index::
   single: spin_compress (keyword in nep.in)

:attr:`spin_compress`
======================

This keyword sets the shared latent radial rank of the spin descriptor.

Syntax
------

::

  spin_compress <count>

Parameters
----------

The default is 3 and the input range is 1--9. The symbol :math:`C` denotes
this value in :ref:`nep_spin`. Each channel is a learned combination of the
structural angular radial channels. It controls descriptor size and cost;
it does not select a fixed list of Hamiltonian terms.
The resulting dimension also depends on :ref:`spin_order <kw_spin_order>`,
:ref:`spin_l_max <kw_spin_l_max>`, and :ref:`spin_soc <kw_spin_soc>`; see :ref:`nep_spin_dimensions`.

Output
------

None.

Examples
--------

::

  spin_compress 3

Notes
-----

None.
