.. _active_xyz:
.. index::
   single: active.xyz (output file)

``active.xyz``
================

File containing atomistic positions, velocities and forces for structures written during on-the-fly active learning.
It is generated when invoking the :ref:`active <kw_active>` keyword.
Only structures with uncertainty exceeding the threshold :math:`\delta` will be written; thus, if no such structure is encountered during the MD simulation, this file will be missing.

File format
-----------
This file is in the `extended XYZ format <https://github.com/libAtoms/extxyz>`_.
The output mode for this file is append.
