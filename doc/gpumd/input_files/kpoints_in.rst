.. _kpoints_in:
.. index::
   single: kpoints.in

k-points (``kpoints.in``)
=========================

This file is used to specify the k points needed for phonon calculations.

File format
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

