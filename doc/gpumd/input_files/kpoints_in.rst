.. _kpoints_in:
.. index::
   single: kpoints.in

k-points (``kpoints.in``)
=========================

This file is used to specify the k points needed for phonon calculations.

File format 1 (before version 5.0)
-----------

The format of this file must be as follows:

.. code::

 N_kpoints
 kx(0) ky(0) kz(0)
 kx(1) ky(1) kz(1)
 ...
 kx(N_kpoints-1) ky(N_kpoints-1) kz(N_kpoints-1)
 
Here,
* ``N_kpoints`` is the number of k points you want to consider.
* The remaining lines give the k vectors (in units of 1/Å) you want to consider.
* The user has to make sure that the k vectors are defined in the reciprocal space with respect to the unit cell chosen.


File format 2 (starting from version 5.0)
-----------

The format of this file must be as follows:

.. code::

 0 0 0 G
 0.5 0 0 M
 0.333 0.333 0 K
 0 0 0 G
 
.. code::

 0 0 0 G
 0.5 0 0.5 X
 0.625 0.25 0.625 U
          #Represents a breakpoint, marking the start of the second path.
 0.375 0.375 0.75 K
 0 0 0 G
 0.5 0.5 0.5 L
 0.5 0.25 0.75 W
 0.5 0 0.5 X

Here,
* The first three columns represent the coordinates of high symmetry points.The last column corresponds to names.
* A blank line indicates a breakpoint or the initiation of a second path.