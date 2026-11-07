.. _dump_xyz:
.. index::
   single: dump.xyz (output file)

``dump.xyz``
===============

File containing per-atom data written by the :ref:`dump_xyz keyword <kw_dump_xyz>`.

The filename is user-defined; ``dump.xyz`` is a common choice.

Every frame contains species and coordinates in ``pos:R:3``. Coordinates are wrapped by default.
``position unwrapped`` writes unwrapped coordinates in ``pos`` and adds ``position_mode=unwrapped``
to the header. The :attr:`unwrapped_position` property instead adds three extra columns alongside the wrapped coordinates.

Additional per-atom properties are included when requested:
:attr:`mass`, :attr:`charge`, :attr:`bec`, :attr:`velocity`, :attr:`force`, :attr:`potential`,
:attr:`unwrapped_position`, :attr:`virial`, :attr:`group_labels`, :attr:`spin`, :attr:`mforce`,
and :attr:`spin_velocity` (written as ``spin_vel``).
The comment line of each frame always carries the time, the periodic boundary flags, the cell,
and the total energy, virial, and stress.

File format
-----------
This file is in the `extended XYZ format <https://github.com/libAtoms/extxyz>`_.

* By default, frames are appended to the specified file.
* With ``split``, frames go into numbered files; each run starts new files, skipping existing ones.
* A trailing ``*`` writes one file per frame, replacing ``*`` with the step number.
