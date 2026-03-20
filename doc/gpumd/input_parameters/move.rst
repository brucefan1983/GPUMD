.. _kw_move:
.. index::
   single: move (keyword in run.in)

:attr:`move`
============

This keyword is used to move part of the system with a constant velocity.

Syntax
------

The :attr:`move` keyword accepts 4 or 5 parameters::

  move <moving_group_id> <velocity_x> <velocity_y> <velocity_z>
  move <grouping_method> <moving_group_id> <velocity_x> <velocity_y> <velocity_z>

- If only :attr:`moving_group_id` and velocities are given, grouping method 0 is used by default.
- If :attr:`grouping_method` is also specified, the given grouping method defined in the :ref:`simulation model file <model_xyz>` will be used.

The last three parameters specify the moving velocity vector, in units of Ångstrom/fs.


Example
-------

One can first equilibrate the system and then move one group of atoms and at the same time fix another group of atoms::

  # equilibration stage
  ensemble npt_scr 300 300 100 0 0 0 100 100 100 1000
  run 1000000

  # production stage
  ensemble nvt_ber 300 300 100
  fix  0            # fix atoms in group 0 (default grouping method 0)
  move 1 0.001 0 0  # move atoms in group 1, with a speed of 0.001 Ångstrom/fs in the x direction (default grouping method 0)
  run  1000000

Using a specific grouping method::

  ensemble nvt_ber 300 300 100
  fix  1 0            # fix group 0 in grouping method 1
  move 1 1 0.001 0 0  # move group 1 in grouping method 1, with a speed of 0.001 Ångstrom/fs in the x direction
  run  1000000

Caveats
-------
* One cannot use NPT when using this keyword. Only ``nvt_ber``, ``nvt_nhc``, ``nvt_bdp``, and ``heat_lan`` ensembles are supported.
* When both :ref:`fix <kw_fix>` and :attr:`move` are used, they must use the same grouping method.
