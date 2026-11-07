.. _dump_xyz:
.. index::
   single: dump.xyz (output file)

``dump.xyz``
===============

File containing per-atom data written by the :ref:`dump_xyz keyword <kw_dump_xyz>`.

The output file name is specified in the ``dump_xyz`` command.

Every frame contains atomic species and positions. The ``pos:R:3`` columns contain wrapped positions by default.
``position unwrapped`` writes only unwrapped positions in ``pos:R:3`` and adds ``position_mode=unwrapped`` to the frame header.
``position both`` writes wrapped positions in ``pos:R:3`` and unwrapped positions in ``unwrapped_position:R:3``.

The following per-atom quantities are included when the corresponding keyword is given:
:attr:`mass`, :attr:`charge`, :attr:`bec`, :attr:`velocity`, :attr:`force`, :attr:`potential`,
:attr:`virial`, :attr:`group_labels`, :attr:`spin`, :attr:`mforce`, and :attr:`spin_velocity`.

The comment line of each frame contains the time, periodic boundary flags, cell,
and the potential energy, virial, and stress of the whole system.

File format
-----------
This file is in the `extended XYZ format <https://github.com/libAtoms/extxyz>`_.

* By default, frames are appended to the file.
* With ``split``, frames are written to numbered files. Existing numbered files are skipped.
* With a filename ending in ``*``, one file is written per frame, replacing ``*`` with the step number within the current run.
  Existing files with the same name are overwritten.
