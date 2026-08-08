.. _kw_dump_netcdf:
.. index::
   single: dump_netcdf (keyword in run.in)

:attr:`dump_netcdf`
===================

Write per-atom data to a user-specified NetCDF trajectory file based on the
`AMBER 1.0 conventions <http://ambermd.org/netcdf/nctraj.pdf>`_.


Syntax
------

This keyword has the following format::

  dump_netcdf <interval> <filename> [{optional_args}]

The :attr:`interval` parameter is the output interval (number of steps) of the atom positions.
:attr:`filename` is the relative or absolute path of the output file.
The optional arguments (:attr:`optional_args`) provide additional functionality.
Currently, the following optional arguments are accepted:

* :attr:`group <grouping_method> <group_id>`

  Write only the atoms of one group.
  :attr:`grouping_method` selects one of the grouping methods defined by
  ``group:I:number_of_grouping_methods`` in the :ref:`simulation model file <model_xyz>`.
  Grouping methods are numbered from 0.
  :attr:`group_id` selects a group label within that grouping method.
  Both values must identify an existing, non-empty group.
  Without this option all atoms in the system are written.

* Per-atom quantities

  Any number of the names below can be given, in any order, each adding one quantity to the
  output.
  The atom types and the wrapped positions are always written, as the ``type`` and
  ``coordinates`` variables.

  .. list-table::
     :header-rows: 1
     :widths: 22 26 34 18

     * - Argument
       - Variable
       - Dimensions
       - Unit
     * - :attr:`velocity`
       - ``velocities``
       - ``(frame, atom, spatial)``
       - Å/ps
     * - :attr:`force`
       - ``forces``
       - ``(frame, atom, spatial)``
       - eV/Å
     * - :attr:`unwrapped_position`
       - ``unwrapped_coordinates``
       - ``(frame, atom, spatial)``
       - Å
     * - :attr:`potential`
       - ``potential_energy``
       - ``(frame, atom)``
       - eV
     * - :attr:`charge`
       - ``charge``
       - ``(frame, atom)``
       - e
     * - :attr:`bec`
       - ``bec``
       - ``(frame, atom, spatial, spatial)``
       - e
     * - :attr:`virial`
       - ``virial``
       - ``(frame, atom, spatial, spatial)``
       - eV
     * - :attr:`mass`
       - ``mass``
       - ``(atom)``
       - amu
     * - :attr:`group_labels`
       - ``group_labels``
       - ``(atom, grouping_method)``
       - --

  The mass and the group labels cannot change during a run and are therefore written once,
  without a frame dimension.
  The ``group_labels`` variable holds one label per grouping method, giving the group each atom
  belongs to.
  The rank-2 tensors are stored row-major, so ``virial[frame, atom, i, j]`` is the *ij*
  component.

* :attr:`precision <value>`

  * If :attr:`value` is ``single``, the output data are 32-bit floating point numbers.
  * If :attr:`value` is ``double``, the output data are 64-bit floating point numbers.

  The default value is ``single``.

* :attr:`compression none|deflate <level>`

  * ``none`` writes an uncompressed NetCDF file in the 64-bit-offset (CDF-2) format and is the default.
    Here, 64-bit offset refers to the file layout, not the precision of the coordinates and velocities;
    these data use 32-bit floating point numbers when the default ``precision single`` is used.
  * ``deflate`` writes a NetCDF4/HDF5 file using lossless compression and requires NetCDF-C built
    with NetCDF4/HDF5 and zlib support. :attr:`level` must be an integer from 0 to 9.
    Level 0 applies no compression. For large, frequently written trajectories, use ``none`` for
    maximum write speed, ``deflate 1`` for a practical balance, or ``deflate 9`` when minimizing
    file size is more important than write speed.
    GPUMD itself does not require NetCDF4/HDF5 support unless ``deflate`` is used.
    If the linked NetCDF-C library was built without NetCDF4/HDF5 support, requesting
    ``deflate`` causes the underlying ``nc_create`` call to fail when GPUMD asks for the
    NetCDF4 file format.
    GPUMD already checks for this and reports a clear error instead of silently writing an
    uncompressed file.
    See :ref:`NetCDF setup <netcdf_setup>` for how to build NetCDF-C with NetCDF4/HDF5 support.

Requirements and specifications
-------------------------------

* This keyword requires an external package to operate.
  Instructions for how to set up the `NetCDF package <https://www.unidata.ucar.edu/software/netcdf>`_ can be found :ref:`here <netcdf_setup>`.
* The ``single`` option is good for saving space and is the default.
* NetCDF output files can be read for example by `VMD <https://www.ks.uiuc.edu/Research/vmd/>`_ or `OVITO <https://ovito.org/>`_ for visualization.
* The NetCDF files also contain atom types, cell lengths, and angles, which can be used in visualization and analysis software.
* The atomic positions are always included in the output. For periodic MD, the wrapped
  positions are written.
* AMBER NetCDF represents a periodic cell using lengths and angles in a standard orientation.
  If the GPUMD cell has a different orientation, the output is rigidly transformed with the cell
  so that periodic geometry is preserved. Positions, velocities, forces and unwrapped positions
  are rotated as vectors, and the per-atom virial and the :term:`BEC` as rank-2 tensors, so that
  every quantity in a frame refers to the same axes.
* The units of the ``forces``, ``potential_energy``, ``charge``, ``bec``, ``virial`` and ``mass``
  variables are those GPUMD works in, and are given by the ``units`` attribute of each variable.
  They are not the units the AMBER conventions specify for the variables AMBER defines.
* For qNEP models the charge values are predicted by the model. For other models they are those
  specified in :ref:`model.xyz <model_xyz>` via ``charge:R:1``.
* The Born effective charge (:term:`BEC`) requires a qNEP model trained with target :term:`BEC`,
  and is refused for any other model.
* :attr:`group_labels` requires at least one grouping method to be defined in
  :ref:`model.xyz <model_xyz>`.

Examples
--------

Single precision without velocities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To dump the whole-system positions every 1000 steps using the default single precision and no compression, one can add::

  dump_netcdf 1000 movie.nc

before the :ref:`run command <kw_run>`.

Double precision with velocities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To dump the whole-system positions and velocities every 1000 steps with 64-bit floating point values, one can add::

  dump_netcdf 1000 movie.nc velocity precision double

before the :ref:`run command <kw_run>`.

Group output with compression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To dump group 0 from grouping method 1 with lossless deflate compression, one can add::

  dump_netcdf 15 group.nc group 1 0 velocity compression deflate 1

before the :ref:`run command <kw_run>`.

Forces and per-atom virials
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To dump forces, per-atom potential energies and per-atom virials every 100 steps with lossless deflate compression, one can add::

  dump_netcdf 100 properties.nc force potential virial compression deflate 1

before the :ref:`run command <kw_run>`.

Multiple groups in one run
^^^^^^^^^^^^^^^^^^^^^^^^^^

Multiple :attr:`dump_netcdf` commands can write different groups during the
same MD run. Each command must use a distinct filename and can use its own
output interval, so every output file contains one independently sampled group::

  dump_netcdf 10 group_0.nc group 0 0 velocity compression deflate 1
  dump_netcdf 25 group_1.nc group 0 1 velocity compression deflate 1

  run 100000


Caveats
-------

* Length is in units of Ångström and velocity is in units of Ångström/picosecond.
* This keyword is not propagating.
  That means, its effect will not be passed from one run to the next.
* This keyword can be invoked multiple times within one run to write different
  groups to different files. Each invocation must use a distinct filename and
  can use its own output interval and options.
* An existing output file is overwritten the first time its name is used in a GPUMD execution.
* Repeating the keyword with the same filename in later runs of the same GPUMD execution
  appends to that file. A different filename creates a separate trajectory file.
* Group, quantity, precision, and compression settings cannot change while appending to the same file.
