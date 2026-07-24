.. _kw_dump_netcdf:
.. index::
   single: dump_netcdf (keyword in run.in)

:attr:`dump_netcdf`
===================

Write the atomic positions (coordinates) and (optionally) velocities to a user-specified
NetCDF trajectory file based on the `AMBER 1.0 conventions <http://ambermd.org/netcdf/nctraj.pdf>`_.


Syntax
------

This keyword has the following format::

  dump_netcdf <grouping_method> <group_id> <interval> <has_velocity> <filename> [{optional_args}]

The :attr:`grouping_method` parameter selects one of the grouping methods defined by
``group:I:number_of_grouping_methods`` in the :ref:`simulation model file <model_xyz>`.
Grouping methods are numbered from 0. When :attr:`grouping_method` is non-negative,
:attr:`group_id` selects a group label within that grouping method, and only the atoms
belonging to that group are written. Both values must identify an existing, non-empty
group. When :attr:`grouping_method` is negative, :attr:`group_id` is ignored and all atoms
in the system are written. These selection rules are the same as for
:ref:`dump_xyz <kw_dump_xyz>`.
The :attr:`interval` parameter is the output interval (number of steps) of the atom positions. :attr:`has_velocity` can be 1 or 0, which means the velocities will or will not be included in the output. :attr:`filename` is the relative or absolute path of the output file.
The optional arguments (:attr:`optional_args`) provide additional functionality.
Currently, the following optional arguments are accepted:

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
  If the GPUMD cell has a different orientation, the output positions and velocities are
  rigidly transformed with the cell so that periodic geometry is preserved.

Examples
--------

Single precision without velocities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To dump the whole-system positions every 1000 steps using the default single precision and no compression, one can add::

  dump_netcdf -1 0 1000 0 movie.nc

before the :ref:`run command <kw_run>`.

Double precision with velocities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To dump the whole-system positions and velocities every 1000 steps with 64-bit floating point values, one can add::

  dump_netcdf -1 0 1000 1 movie.nc precision double

before the :ref:`run command <kw_run>`.

Group output with compression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To dump group 0 from grouping method 1 with lossless deflate compression, one can add::

  dump_netcdf 1 0 15 1 group.nc compression deflate 1

before the :ref:`run command <kw_run>`.


Caveats
-------

* Length is in units of Ångström and velocity is in units of Ångström/picosecond.
* This keyword is not propagating.
  That means, its effect will not be passed from one run to the next.
* An existing output file is overwritten the first time its name is used in a GPUMD execution.
* Repeating the keyword with the same filename in later runs of the same GPUMD execution
  appends to that file. A different filename creates a separate trajectory file.
* Group, precision, velocity, and compression settings cannot change while appending to the same file.
* The new syntax is not compatible with the former ``dump_netcdf <interval> <has_velocity>`` syntax.
