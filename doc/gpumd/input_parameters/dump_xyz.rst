.. _kw_dump_xyz:
.. index::
   single: dump_xyz (keyword in run.in)

:attr:`dump_xyz`
================

Write per-atom data in `extended XYZ format <https://github.com/libAtoms/extxyz>`_.

Syntax
------

.. code::

   dump_xyz <interval> <filename> [group <grouping_method> <group_id>] [position <wrapped|unwrapped>] [precision <single|double>] [split <frames_per_file>] {<property_1> <property_2> ...}

* :attr:`interval`: output interval in MD steps (a positive integer).
* :attr:`filename`: output file. A trailing ``*`` writes one file per frame, replacing ``*`` with the step number.
* :attr:`group`: select atoms from the given grouping method and group ID. The default is the whole system.
  The selection applies to coordinates and all requested properties.
* :attr:`position`: select ``wrapped`` (default) or ``unwrapped`` coordinates for the ``pos:R:3`` columns.
* :attr:`precision`: significant digits for all floating-point values.
  ``single`` (default) writes nine digits; ``double`` writes 17. These preserve single- and double-precision values, respectively.
* :attr:`split`: maximum frames per file (a positive integer).
  For example, ``split 10000`` with ``trajectory.xyz`` writes ``trajectory_0001.xyz``, ``trajectory_0002.xyz``, etc.
  The last file may contain fewer frames. Existing numbered files are skipped.
  This option cannot be combined with a trailing ``*``.

Species and coordinates are always included. Optional properties are :attr:`mass`, :attr:`velocity`,
:attr:`force`, :attr:`potential`, :attr:`virial`, :attr:`charge`, :attr:`bec`, :attr:`group_labels`,
:attr:`spin`, :attr:`mforce`, :attr:`spin_velocity`, and :attr:`unwrapped_position`.
The :attr:`group_labels` property adds one integer column per grouping method with each atom's group ID.


Coordinate output
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Coordinate option
     - Coordinates written
   * - None (default), or ``position wrapped``
     - Wrapped coordinates in ``pos``.
   * - ``position unwrapped``
     - Only unwrapped coordinates, in ``pos``.
   * - ``unwrapped_position``
     - Wrapped coordinates in ``pos``; unwrapped coordinates in ``unwrapped_position``.

Use ``position unwrapped`` to save space by storing only one coordinate set.
GPUMD marks these frames with ``position_mode=unwrapped`` in the header; this is not a ``run.in`` option.
``position unwrapped`` and ``unwrapped_position`` cannot be combined.

Unwrapped coordinates preserve continuity across periodic boundaries and may lie outside the simulation cell.
This applies to orthogonal and triclinic cells. Nonperiodic directions are not wrapped.


Examples
--------

.. code::

    ensemble xxx # some ensemble

    # wrapped positions:
    dump_xyz 1000 positions.xyz

    # unwrapped positions and velocities:
    dump_xyz 15 dump.xyz position unwrapped velocity

    # both coordinate sets and velocities:
    dump_xyz 15 both.xyz unwrapped_position velocity

    # group 0 of method 1:
    dump_xyz 15 group.xyz group 1 0 position unwrapped velocity

    # additional properties for group 0 of method 1:
    dump_xyz 100 properties.xyz group 1 0 mass velocity potential force virial

    # double-precision forces:
    dump_xyz 100 forces.xyz precision double force

    # 10000 frames per file:
    dump_xyz 100 trajectory.xyz split 10000

    run 1000000

Caveats
-------
* Repeat this keyword for each ``run``; it does not carry over between runs.
* With no ``split`` option or trailing ``*``, frames are appended to the output file.
  With ``split``, each run creates new numbered files without overwriting existing chunks.
* Multiple ``dump_xyz`` commands can be used in one run.
* For qNEP models, the charge values dumped out are predicted by the qNEP models.
  For other models, the charge values are those specified in :attr:`model.xyz` via :attr:`charge:R:1`.
* The Born effective charge (:term:`BEC`) is only meaningful for a qNEP model trained with target :term:`BEC`.
* ``spin`` and ``mforce`` are three-component vectors that require both a
  NEP-spin model and ``spin:R:3`` in ``model.xyz``. The XYZ property alone
  does not enable these outputs for other potential models.
* The per-atom ``virial`` property contains all nine row-major components
  ``xx xy xz yx yy yz zx zy zz``. NEP_Spin does not symmetrize this per-atom
  tensor.

The ``spin_velocity`` option writes ``spin_vel:R:3`` in
:math:`\mu_B/\mathrm{fs}`. It requires a NEP-spin model, ``spin:R:3``, and
velocities initialized by TSPIN or supplied as ``spin_vel:R:3`` in
``model.xyz``. Ordinary models ignore additional spin input properties.
