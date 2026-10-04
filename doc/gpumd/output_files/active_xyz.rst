.. _active_xyz:
.. index::
   single: active.xyz (output file)

``active.xyz``
================

File containing atomistic positions, velocities, forces and uncertainties for structures written during on-the-fly active learning.
It is generated when invoking the :ref:`active <kw_active>` keyword.
A structure is written when its uncertainty exceeds the threshold :math:`\delta` or is NaN.
If no structure is written during a run, the run adds nothing to this file.

File format
-----------
This file is in the `extended XYZ format <https://github.com/libAtoms/extxyz>`_.
The output mode for this file is append.
