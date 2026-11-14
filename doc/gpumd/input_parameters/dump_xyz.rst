.. _kw_dump_xyz:
.. index::
   single: dump_xyz (keyword in run.in)

:attr:`dump_xyz`
================

Write per-atom data into user-specified file(s) in `extended XYZ format <https://github.com/libAtoms/extxyz>`_.

Runs using :ref:`multi_replica <kw_multi_replica>` use the :ref:`replica trajectory syntax <replica_kw_dump_xyz>` instead of the ordinary MD syntax below.

Syntax
------

.. code::

   dump_xyz <interval> <filename> [group <grouping_method> <group_id>] [precision <single|double>] [split <frames_per_file>] {<property_1> <property_2> ...}

Parameters
----------

* :attr:`interval` is the output interval (number of steps) of the data.

* :attr:`filename` is the output file.

  If it ends in ``*``, each frame is written to a separate file, with the star replaced by the step number.

* The optional :attr:`split` parameter is the maximum number of frames written per XYZ file (a positive integer).
  For example, ``dump_xyz 100 trajectory.xyz split 10000`` writes 10000 frames to ``trajectory_0001.xyz``, then 10000 frames to ``trajectory_0002.xyz``, and so on.
  The final file can contain fewer frames. Existing numbered files are skipped to avoid overwriting data when restarting or issuing another ``run`` command.
  The :attr:`split` option cannot be used with a filename ending in ``*``.

* The :attr:`group` option restricts the output to the atoms in group :attr:`group_id` of grouping method :attr:`grouping_method`.

  Without this option, all atoms are written.

* The :attr:`precision` option controls the number of significant digits written for every floating-point value.

  ``single`` (default) writes nine significant digits; ``double`` writes 17.
  These preserve single- and double-precision values, respectively.

* Then one can write the properties to be output, and the allowed properties include: :attr:`mass`, :attr:`velocity`, :attr:`force`, :attr:`potential`, :attr:`virial`, :attr:`charge`, :attr:`bec`, :attr:`group_labels`, :attr:`spin`, :attr:`mforce`, :attr:`unwrapped_position`, and :attr:`unwrapped_position_only`.

  ``group_labels`` writes one integer column per grouping method, containing each atom's group label.

* The wrapped positions are included by default.
  The :attr:`unwrapped_position` option writes both wrapped and unwrapped positions.
  The :attr:`unwrapped_position_only` option writes only unwrapped positions.
  The :attr:`unwrapped_position` and :attr:`unwrapped_position_only` options are mutually exclusive.


Examples
--------

.. code::

    # dump positions every 1000 steps, for the whole system:
    dump_xyz 1000 positions.xyz

    # dump wrapped and unwrapped positions, plus velocities:
    dump_xyz 15 both.xyz unwrapped_position velocity

    # dump only unwrapped positions, plus velocities:
    dump_xyz 15 dump.xyz unwrapped_position_only velocity

    # dump many other quantities every 100 steps, for atoms in group 0 of grouping method 1:
    dump_xyz 100 properties.xyz group 1 0 mass velocity potential force virial

    # dump forces with the full precision of the underlying double-precision values:
    dump_xyz 100 forces.xyz precision double force

    # dump 10000 frames per file, with one frame every 100 steps:
    dump_xyz 100 trajectory.xyz split 10000

    run 1000000

Notes
-----
* Repeat this command before each run that needs this output.
* With no ``split`` option or trailing ``*``, the output file has an appending behavior.
  With ``split``, each run creates new numbered files without overwriting existing chunks.
* Different from many of the other keywords, this keyword is allowed to be invoked multiple times within one run.
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
