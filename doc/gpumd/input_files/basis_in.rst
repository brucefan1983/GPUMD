.. _basis_in:
.. index::
   single: basis.in

Primitive cell (``basis.in``)
=============================

This file is used to define the unit cell for phonon calculations.

File format
-----------

The format of this file must be as follows:

.. code::

 N_basis
 id(0) mass(0)
 id(1) mass(1)
 ...
 id(N_basis-1) mass(N_basis-1)
 map(0)
 map(1)
 ...
 map(N-1)
 
Here,
* ``N_basis`` is the number of atoms in the unit cell you choose. 
For example, it can be 2 for diamond silicon if you use the primitive cell as the unit cell.
* The next ``N_basis`` lines contain the atom indices (using the order as in the simulation model file; starting from 0) 
and masses for the basis atoms.
* The remaining ``N`` lines map the ``N`` atoms in the simulation model to the basis atoms. 
If the n-th atom in the simulatoin model file is equivalent to (under translation) the m-th basis atom in the unit cell, 
we have map(n)=m.
