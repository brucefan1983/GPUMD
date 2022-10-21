.. _kw_dump_netcdf:
.. index::
   single: dump_netcdf (keyword in run.in)

:attr:`dump_netcdf`
===================

Write the atomic positions (coordinates) in NetCDF format to a `movie.nc file <http://ambermd.org/netcdf/nctraj.pdf>`_.


Syntax
------

This keyword has the following format::

  dump_netcdf interval <optional_args>

The :attr:`interval` parameter is the output interval (number of steps) of the atom positions.
The :attr:`optional_args` provide additional functionality.
Currently, the following optional argument is accepted:

* :attr:`precision`
  
  * If :attr:`precision` is ``single``, the output data are 32-bit floating point numbers.
  * If :attr:`precision` is ``double``, the output data are 64-bit floating point numbers.

  The default value is ``double``.

Requirements and specifications
-------------------------------

* This keyword requires an external package to operate.
  Instructions for how to set up the `NetCDF package <https://www.unidata.ucar.edu/software/netcdf>`_ can be found :ref:`here <netcdf_setup>`.
* The NetCDF format follows the `AMBER specifications <http://ambermd.org/netcdf/nctraj.pdf>`_. 
* The ``single`` option is good for saving space, but does not follow the official AMBER 1.0 conventions.
* NetCDF output files can be read for example by `VMD <https://www.ks.uiuc.edu/Research/vmd/>`_ or `OVITO <https://ovito.org/>`_ for visualization. 
* The NetCDF files also contain cell lengths and angles, which can be used in the visualization software to illustrate boundaries and show periodic copies of the structure.

Examples
--------

Single precision example
^^^^^^^^^^^^^^^^^^^^^^^^

To dump the positions every 1000 steps to a NetCDF file with 32-bit floating point values, one can add::

  dump_netcdf 1000 precision single

before the :ref:`run command <kw_run>`.

Double precision example
^^^^^^^^^^^^^^^^^^^^^^^^

To dump the positions every 1000 steps to a NetCDF file with 64-bit floating point values, one can add::

  dump_netcdf 1000

before the :ref:`run command <kw_run>`.


Caveats
-------

* This keyword is not propagating.
  That means, its effect will not be passed from one run to the next.
* The output appends to the same file for different runs in the same simulation.
  Re-running the simulation will create a new output file.
* If the :attr:`precision` changes between different runs, the first defined precision will still be used (i.e., changes in precision are ignored during a simulation). 
