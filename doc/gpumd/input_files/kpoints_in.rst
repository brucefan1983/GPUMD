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
# high symmetry points
points_x(0) points_y(0) points_z(0) name(0)
points_x(1) points_y(1) points_z(1) name(1)
...


Example
-------

For example, the command::

.. code::

 0 0 0 G
 0.5 0 0 M
 0.333 0.333 0 K
 0 0 0 G
 
or

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