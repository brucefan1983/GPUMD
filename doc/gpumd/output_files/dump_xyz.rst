.. _dump_xyz:
.. index::
   single: dump.xyz (output file)

``dump.xyz``
===============

File containing per-atom data written by the :ref:`dump_xyz keyword <kw_dump_xyz>`.

The file name is chosen by the user, so ``dump.xyz`` is only a common convention rather than a
fixed name.
If the name given to :ref:`dump_xyz <kw_dump_xyz>` ends with a star (``*``), one file is written
per frame and the star is replaced by the step number.

Every frame contains the atomic species and positions.
The ``pos:R:3`` columns contain wrapped positions by default.
The ``unwrapped_position`` option keeps wrapped positions in ``pos:R:3`` and adds
unwrapped positions in ``unwrapped_position:R:3``.
The ``unwrapped_position_only`` option writes only unwrapped positions in ``pos:R:3``
and adds ``position_mode=unwrapped`` to the frame header.

The following per-atom quantities are included when the corresponding keyword is given:
``mass``, ``charge``, ``bec``, ``velocity``, ``force``, ``potential``,
``unwrapped_position``, ``virial``, and ``group_labels``.
The comment line of each frame always carries the time, the periodic boundary flags, the cell,
and the total energy, virial, and stress.
With a NEP-Spin potential, the header field ``virial`` sums the raw per-atom
virials over the whole system in eV, in row-major order
``xx xy xz yx yy yz zx zy zz``.
The three pairs of off-diagonal entries can differ, even when the trajectory
contains only a group or omits per-atom virials.
In spin NPT, the separate header field ``stress`` contains the pressure
computed from the symmetric virial and the lattice kinetic term.

File format
-----------
This file is in the `extended XYZ format <https://github.com/libAtoms/extxyz>`_.
The output mode for this file is append.
