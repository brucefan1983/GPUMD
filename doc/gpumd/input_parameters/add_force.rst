.. _kw_add_force:
.. index::
   single: add_force (keyword in run.in)

``add_force``
=============

Add an external force to each atom in a selected group during a run.

Syntax
------

::

  add_force <group_method> <group_id> <Fx> <Fy> <Fz>
  add_force <group_method> <group_id> <add_force_file>

Parameters
----------

* ``group_method`` and ``group_id``: Select a group defined in :ref:`model.xyz <grouping_methods>`.
* ``Fx``, ``Fy``, and ``Fz``: Cartesian components of a constant force, in eV/Å, applied to each selected atom.
* ``add_force_file``: File containing a sequence of force vectors, in eV/Å, applied periodically to each selected atom.
  Its format is::

    N
    Fx(0) Fy(0) Fz(0)
    Fx(1) Fy(1) Fz(1)
    ...
    Fx(N-1) Fy(N-1) Fz(N-1)

  The first line gives the number of vectors, :math:`N\geq2`, followed by one vector per line.
  Vector 0 is used at the initial force evaluation (:math:`t=0`).
  At :math:`t=n\Delta t`, vector :math:`n\bmod N` is used.

Examples
--------

Apply a constant force of 0.1 eV/Å along x to each atom in group 1 of grouping method 2::

  add_force 2 1 0.1 0 0

Apply a repeating force sequence to group 2 of grouping method 0::

  add_force 0 2 add_force.txt

For example, ``add_force.txt`` can contain three vectors::

  3
  0.10 0.00 0.00
  0.00 0.10 0.00
  0.00 0.00 0.10

Notes
-----

This command can be used multiple times for one run.
Repeat it before each subsequent run that requires the force.
