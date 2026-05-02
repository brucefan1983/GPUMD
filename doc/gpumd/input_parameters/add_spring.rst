.. _kw_add_spring:
.. index::
   single: add_spring (keyword in run.in)

:attr:`add_spring`
==================

This keyword adds spring interactions to selected atom groups at each step of a run.

It supports three modes: :attr:`ghost_com`, :attr:`ghost_atom`, and :attr:`com_com`, which add spring forces between a ghost atom and the center of mass (COM) of a group, between ghost atoms and their corresponding atoms in a group, and between the COMs of two groups, respectively. The ghost atom(s) can be static or move at a constant velocity. Each mode supports two spring types: :attr:`couple` (a single radial spring) and :attr:`decouple` (three independent springs along the x, y, and z directions).

Syntax
------

The complete syntax for the six combinations is::

  add_spring ghost_com <group_method> <group_id> <ghost_vx> <ghost_vy> <ghost_vz> couple <k_couple> <R0> <offset_x> <offset_y> <offset_z>
  add_spring ghost_com <group_method> <group_id> <ghost_vx> <ghost_vy> <ghost_vz> decouple <k_decouple_x> <k_decouple_y> <k_decouple_z> <offset_x> <offset_y> <offset_z>
  add_spring ghost_atom <group_method> <group_id> <ghost_vx> <ghost_vy> <ghost_vz> couple <k_couple> <R0> <offset_x> <offset_y> <offset_z>
  add_spring ghost_atom <group_method> <group_id> <ghost_vx> <ghost_vy> <ghost_vz> decouple <k_decouple_x> <k_decouple_y> <k_decouple_z> <offset_x> <offset_y> <offset_z>
  add_spring com_com <group_method> <group_id_1> <group_id_2> couple <k_couple> <R0>
  add_spring com_com <group_method> <group_id_1> <group_id_2> decouple <k_decouple_x> <k_decouple_y> <k_decouple_z>

The coupled spring potential is:

.. math::

   U_{\mathrm{couple}} = \frac{1}{2} k_{\mathrm{couple}} (R - R_0)^2

where :math:`R` is the distance between the two points connected by the spring.

The decoupled spring potential is:

.. math::

   U_{\mathrm{decouple}} = \frac{1}{2} k_{\mathrm{decouple},x} (x_2 - x_1)^2 + \frac{1}{2} k_{\mathrm{decouple},y} (y_2 - y_1)^2 + \frac{1}{2} k_{\mathrm{decouple},z} (z_2 - z_1)^2

where :math:`(x_1, y_1, z_1)` and :math:`(x_2, y_2, z_2)` are the Cartesian coordinates of the two points connected by the spring.

* In :attr:`ghost_com` mode, force is added to atoms in group :attr:`group_id` of group method :attr:`group_method` with mass-weighted distribution. The ghost is initially placed at the COM + (:attr:`offset_x`, :attr:`offset_y`, :attr:`offset_z`) and is displaced by (:attr:`ghost_vx`, :attr:`ghost_vy`, :attr:`ghost_vz`) at each step.
* In :attr:`ghost_atom` mode, each atom in group :attr:`group_id` is attached to its own ghost anchor. The anchor is initially placed at the atom position + (:attr:`offset_x`, :attr:`offset_y`, :attr:`offset_z`) and is displaced by (:attr:`ghost_vx`, :attr:`ghost_vy`, :attr:`ghost_vz`) at each step. The input spring constant(s) (:attr:`k_couple` or :attr:`k_decouple_x`, :attr:`k_decouple_y`, :attr:`k_decouple_z`) represent the total stiffness of the entire group and are divided internally by the number of atoms in the group. Thus, each atom experiences a spring with an effective stiffness of :attr:`k / N_atoms`.
* In :attr:`com_com` mode, a spring interaction is applied between the COM of group :attr:`group_id_1` and the COM of group :attr:`group_id_2` under the same :attr:`group_method`. Equal and opposite forces are applied to the two groups with mass-weighted distribution within each group.
* In :attr:`couple` mode, :attr:`k_couple` must be positive and :attr:`R0` must be non-negative.
* In :attr:`decouple` mode, :attr:`k_decouple_x`, :attr:`k_decouple_y`, and :attr:`k_decouple_z` must be non-negative.
* In :attr:`com_com` mode, :attr:`group_id_1` and :attr:`group_id_2` must be different.
* Force is in units of eV/Å, distance is in units of Å, velocity is in units of Å/step, and spring constant is in units of eV/Å².
* Spring forces are written to the file :attr:`spring_force_*.out`, where the asterisk is replaced by a zero-based index that increments with each invocation of the command. The meaning of each column in the file is::

   # step  mode  Fx  Fy  Fz  Ftotal  energy

where :attr:`mode` is an integer flag: 0 = ghost_com, 1 = ghost_atom, 2 = com_com. In :attr:`com_com` mode, the reported :attr:`Fx`, :attr:`Fy`, and :attr:`Fz` correspond to the net spring force applied to :attr:`group_id_1`.

Example 1 (ghost_com + couple)
------------------------------

Add a coupled spring with spring constant 10 eV/Å² and equilibrium distance 0 Å between a static ghost atom and the COM of atoms in group 2 defined by group method 0. The ghost atom is initially located at the COM::

   add_spring ghost_com 0 2 0 0 0 couple 10 0 0 0 0

Example 2 (ghost_com + decouple)
--------------------------------

Add a decoupled spring with spring constants 10 eV/Å² in the x direction and 0 eV/Å² in the y and z directions between a ghost atom moving at velocity (0.00005, 0, 0) Å/step and the COM of atoms in group 2 defined by group method 0. The ghost atom is initially located at the COM::

   add_spring ghost_com 0 2 0.00005 0 0 decouple 10 0 0 0 0 0

Example 3 (ghost_atom + couple)
-------------------------------

Add a coupled spring between each atom in group 2 defined by group method 0 and its corresponding moving ghost anchor. Each anchor is initially placed at the corresponding atom position and moves at velocity (0.00005, 0, 0) Å/step::

   add_spring ghost_atom 0 2 0.00005 0 0 couple 10 0 0 0 0

Example 4 (com_com + decouple)
------------------------------

Add a decoupled Cartesian spring between the COM of group 1 and the COM of group 2 under the same group method 0::

   add_spring com_com 0 1 2 decouple 10 0 0

Note
----

This keyword can be used multiple times during a run.