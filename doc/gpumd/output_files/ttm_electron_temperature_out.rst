.. _ttm_electron_temperature_out:
.. index::
   single: ttm_electron_temperature.out (output file)

``ttm_electron_temperature.out``
================================

This file contains electron temperature snapshots from :attr:`ttm` and
:attr:`heat_ttm` runs.
It is written every :attr:`ttm_out_interval` steps and the output mode is overwrite.

File format
-----------

The file starts with a header such as::

  # electron temperature snapshots for TTM
  # nx 1 ny 1 nz 12
  # active_x 1 1 active_y 1 1 active_z 3 10
  # properties_file yes
  # electron_source 0.0000000000e+00
  # output_interval 50 step(s)
  # columns: ix iy iz T_e[K]

Each snapshot starts with::

  # step 2000

followed by one line for each electron grid cell::

  ix iy iz T_e

where:

* :attr:`ix`, :attr:`iy`, :attr:`iz` are the 1-based electron-grid indices
* :attr:`T_e` is the electron temperature in K

Notes
-----

* The grid order is x-fastest, then y, then z.
* Cells outside the active TTM region are written with zero electron temperature.
