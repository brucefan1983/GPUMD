.. _kw_add_spring:
.. index::
   single: add_spring (keyword in run.in)

:attr:`add_spring`
==================

This keyword is used to add a spring between a ghost atom and the centroid of atoms in a selected group.

Syntax
------

This keyword is used in one of the following two ways::

  add_spring ghost_com <group_method> <group_id> <ghost_vx> <ghost_vy> <ghost_vz> couple <k_couple> <R0> <offset_x> <offset_y> <offset_z>  # usage 1
  add_spring ghost_com <group_method> <group_id> <ghost_vx> <ghost_vy> <ghost_vz> decouple <k_decouple_x> <k_decouple_y> <k_decouple_z> <offset_x> <offset_y> <offset_z>  # usage 2

The function of the coupled spring potential is given by the equation:

.. math::

   U_{\mathrm{couple}} = \frac{1}{2} k_{\mathrm{couple}} (R - R_0)^2
   
where :math:`R` is the distance between the ghost atom and the centroid of the selected group. The function of the decoupled spring potential is given by the equation:

.. math::

   U_{\mathrm{decouple}} = \frac{1}{2} k_{\mathrm{decouple},x} (x - x_{\mathrm{ghost}})^2 + \frac{1}{2} k_{\mathrm{decouple},y} (y - y_{\mathrm{ghost}})^2 + \frac{1}{2} k_{\mathrm{decouple},z} (z - z_{\mathrm{ghost}})^2
   
where :math:`(x_{\mathrm{ghost}}, y_{\mathrm{ghost}}, z_{\mathrm{ghost}})` is the position of the ghost atom and :math:`(x, y, z)` is the position of the centroid of the selected group.

* Force is added to atoms in group :attr:`group_id` of group method :attr:`group_method`.
* In the first usage, a spring is coupled to the ghost atom with spring constant :attr:`k_couple` and equilibrium distance :attr:`R0`.
* In the second usage, a spring is decoupled from the ghost atom with spring constants :attr:`k_decouple_x`, :attr:`k_decouple_y`, and :attr:`k_decouple_z` in the x, y, and z directions, respectively.
* The ghost atom starts at the position of the centroid of the selected group plus the offset vector (:attr:`offset_x`, :attr:`offset_y`, :attr:`offset_z`), and moves with velocity (:attr:`ghost_vx`, :attr:`ghost_vy`, :attr:`ghost_vz`).
* Force is in units of eV/Å, distance is in units of Å, velocity is in units of Å/step, and spring constant is in units of eV/Å².
* Spring forces are written to `spring_force_*.out`, where the asterisk is replaced by a zero-based index that increments with each invocation of the command. The meaning of each column in the file is::

  # step  call  mode  Fx  Fy  Fz  energy  spring_force


Example 1
---------

Add a coupled spring with spring constant 10 eV/Å² and equilibrium distance 0 Å between a static ghost atom and the centroid of atoms in group 2 of group method 0. The ghost atom is initially in the same position as the centroid::

   add_spring    ghost_com 0 2 0 0 0 couple 10 0 0 0 0

Example 2
---------

Add a decoupled spring with spring constants 10 eV/Å² in the x direction and 0 eV/Å² in the y and z directions between a ghost atom moving with velocity (0.00005, 0, 0) Å/step and the centroid of atoms in group 2 of group method 0. The ghost atom is initially in the same position as the centroid::

   add_spring    ghost_com 0 2 0.00005 0 0 decouple 10 0 0 0 0 0

Note
----

This keyword can be used multiple times during a run.
