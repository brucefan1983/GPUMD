.. _kw_move:
.. index::
   single: move (keyword in run.in)

:attr:`move`
============

This keyword is used to move part of the system with a constant velocity.

Syntax
------

The :attr:`move` keyword requires 4 parameters::

  move <moving_group_id> <velocity_x> <velocity_y> <velocity_z>

Here, :attr:`moving_group_id` specifies the group id for the moving part, which should be defined in the grouping method 0.
The next three parameters specify the moving velocity vector, in units of Ångstrom/fs.


Example
-------

One can first equilibrate the system and then move one group of atoms and at the same time fix another group of atoms::

  # equilibration stage
  ensemble npt_scr 300 300 100 0 0 0 100 100 100 1000
  run 1000000

  # production stage
  ensemble nvt_scr 300 300 100
  fix  0            # fix atoms in group 0
  move 1 0.001 0 0  # move atoms in group 1, with a speed of 0.001 Ångstrom/fs in the x direction
  run  1000000

Caveats
-------
* One cannot use NPT when using this keyword.
* Currently, the moving group must be defined in the grouping method 0. This might be extended in a future version.
