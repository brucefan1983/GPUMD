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

Every frame contains the atomic species and the wrapped positions.
The following per-atom quantities are included when the corresponding keyword is given:
:attr:`mass`, :attr:`charge`, :attr:`bec`, :attr:`velocity`, :attr:`force`, :attr:`potential`,
:attr:`unwrapped_position`, :attr:`virial`, and :attr:`group_labels`.
The comment line of each frame always carries the time, the periodic boundary flags, the cell,
and the total energy, virial, and stress.

File format
-----------
This file is in the `extended XYZ format <https://github.com/libAtoms/extxyz>`_.
The output mode for this file is append.
