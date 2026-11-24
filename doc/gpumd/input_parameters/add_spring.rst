.. _kw_add_spring:
.. index::
   single: add_spring (keyword in run.in)

``add_spring``
==================

This keyword adds spring interactions to selected atom groups at each step of a run.

It supports three modes: ``ghost_com``, ``ghost_atom``, and ``com_com``, which add spring forces between a ghost atom and the center of mass (COM) of a group, between ghost atoms and their corresponding atoms in a group, and between the COMs of two groups, respectively. The ghost atom(s) can be static or move at a constant velocity. Each mode supports two spring types: ``couple`` (a single radial spring) and ``decouple`` (three independent springs along the x, y, and z directions).

Syntax
------

The complete syntax for the six combinations is::

  add_spring ghost_com <group_method> <group_id> <ghost_vx> <ghost_vy> <ghost_vz> couple <k_couple> <R0> <offset_x> <offset_y> <offset_z> [continue <spring_id>]
  add_spring ghost_com <group_method> <group_id> <ghost_vx> <ghost_vy> <ghost_vz> decouple <k_decouple_x> <k_decouple_y> <k_decouple_z> <offset_x> <offset_y> <offset_z> [continue <spring_id>]
  add_spring ghost_atom <group_method> <group_id> <ghost_vx> <ghost_vy> <ghost_vz> couple <k_couple> <R0> <offset_x> <offset_y> <offset_z> [continue <spring_id>]
  add_spring ghost_atom <group_method> <group_id> <ghost_vx> <ghost_vy> <ghost_vz> decouple <k_decouple_x> <k_decouple_y> <k_decouple_z> <offset_x> <offset_y> <offset_z> [continue <spring_id>]
  add_spring com_com <group_method> <group_id_1> <group_id_2> couple <k_couple> <R0>
  add_spring com_com <group_method> <group_id_1> <group_id_2> decouple <k_decouple_x> <k_decouple_y> <k_decouple_z>

Parameters
----------

The coupled spring potential is:

.. math::

   U_{\mathrm{couple}} = \frac{1}{2} k_{\mathrm{couple}} (R - R_0)^2

where :math:`R` is the distance between the two points connected by the spring.

The decoupled spring potential is:

.. math::

   U_{\mathrm{decouple}} = \frac{1}{2} k_{\mathrm{decouple},x} (x_2 - x_1)^2 + \frac{1}{2} k_{\mathrm{decouple},y} (y_2 - y_1)^2 + \frac{1}{2} k_{\mathrm{decouple},z} (z_2 - z_1)^2

where :math:`(x_1, y_1, z_1)` and :math:`(x_2, y_2, z_2)` are the Cartesian coordinates of the two points connected by the spring.

* In ``ghost_com`` mode, force is added to atoms in group ``group_id`` of group method ``group_method`` with mass-weighted distribution. The ghost is initially placed at the COM + (``offset_x``, ``offset_y``, ``offset_z``) and is displaced by (``ghost_vx``, ``ghost_vy``, ``ghost_vz``) at each step.
* In ``ghost_atom`` mode, each atom in group ``group_id`` is attached to its own ghost anchor. The anchor is initially placed at the atom position + (``offset_x``, ``offset_y``, ``offset_z``) and is displaced by (``ghost_vx``, ``ghost_vy``, ``ghost_vz``) at each step. The input spring constant(s) (``k_couple`` or ``k_decouple_x``, ``k_decouple_y``, ``k_decouple_z``) represent the total stiffness of the entire group and are divided internally by the number of atoms in the group. Thus, each atom experiences a spring with an effective stiffness of ``k / N_atoms``.
* In ``com_com`` mode, a spring interaction is applied between the COM of group ``group_id_1`` and the COM of group ``group_id_2`` under the same ``group_method``. Equal and opposite forces are applied to the two groups with mass-weighted distribution within each group.
* In ``couple`` mode, ``k_couple`` must be positive and ``R0`` must be non-negative.
* In ``decouple`` mode, ``k_decouple_x``, ``k_decouple_y``, and ``k_decouple_z`` must be non-negative.
* In ``com_com`` mode, ``group_id_1`` and ``group_id_2`` must be different.
* Force is in units of eV/Å, distance is in units of Å, velocity is in units of Å/step, and spring constant is in units of eV/Å².

Output
------

Spring forces are written to ``spring_gm<group_method>_g<group_id>_s<spring_id>.out``. The ``spring_id`` is assigned independently for each (``group_method``, ``group_id``) pair. A new spring uses the lowest unused non-negative ID that does not already have a spring output or restart file. A continued spring keeps the ID specified by ``continue <spring_id>``. The meaning of each column in the file is::

   # step  mode  Fx  Fy  Fz  Ftotal  energy

where ``mode`` is an integer flag: 0 = ghost_com, 1 = ghost_atom, 2 = com_com. In ``com_com`` mode, the reported ``Fx``, ``Fy``, and ``Fz`` correspond to the net spring force applied to ``group_id_1``.

Examples
--------

Example 1 (ghost_com + couple)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a coupled spring with spring constant 10 eV/Å² and equilibrium distance 0 Å between a static ghost atom and the COM of atoms in group 2 defined by group method 0. The ghost atom is initially located at the COM::

   add_spring ghost_com 0 2 0 0 0 couple 10 0 0 0 0

Example 2 (ghost_com + decouple)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a decoupled spring with spring constants 10 eV/Å² in the x direction and 0 eV/Å² in the y and z directions between a ghost atom moving at velocity (0.00005, 0, 0) Å/step and the COM of atoms in group 2 defined by group method 0. The ghost atom is initially located at the COM::

   add_spring ghost_com 0 2 0.00005 0 0 decouple 10 0 0 0 0 0

Example 3 (ghost_atom + couple)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a coupled spring between each atom in group 2 defined by group method 0 and its corresponding moving ghost anchor. Each anchor is initially placed at the corresponding atom position and moves at velocity (0.00005, 0, 0) Å/step::

   add_spring ghost_atom 0 2 0.00005 0 0 couple 10 0 0 0 0

Example 4 (com_com + decouple)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a decoupled Cartesian spring between the COM of group 1 and the COM of group 2 under the same group method 0::

   add_spring com_com 0 1 2 decouple 10 0 0

Notes
-----

This keyword can be used multiple times during a run.

For ``ghost_com`` and ``ghost_atom``, the ghost state is automatically saved at the end of each run to ``spring_gm<group_method>_g<group_id>_s<spring_id>.restart``. To continue a ghost spring, append ``continue <spring_id>`` to the command. The restart is selected by the current group method, group ID, and the specified spring ID. For example::

   add_spring ghost_com 0 2 0.00005 0 0 decouple 10 0 0 0 0 0   # spring ID 0
   add_spring ghost_com 0 2 0.00010 0 0 decouple 10 0 0 0 0 0   # spring ID 1
   add_spring ghost_com 0 2 0.00015 0 0 decouple 10 0 0 0 0 0   # spring ID 2
   run 100000

   add_spring ghost_com 0 2 0.00005 0 0 decouple 10 0 0 0 0 0 continue 0
   add_spring ghost_com 0 2 0.00015 0 0 decouple 10 0 0 0 0 0 continue 2
   run 100000

The continued spring must use the same ghost mode, group method, group ID, spring ID, and group size. Other loading parameters, such as velocity and spring constants, may be changed. The offset is not reapplied when ``continue`` is used. The same spring ID cannot be used more than once for the same group within one run. The restart file is retained while the new run is executing and is replaced after that run completes successfully. ``com_com`` does not support ``continue``.
