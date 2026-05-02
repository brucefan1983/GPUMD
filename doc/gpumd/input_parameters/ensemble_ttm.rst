.. _kw_ensemble_ttm:

:attr:`ensemble` (TTM)
======================

This page describes the two-temperature-model (TTM) integrators
:attr:`ttm` and :attr:`heat_ttm`.

Syntax
------

:attr:`ttm`
^^^^^^^^^^^
The full command is::

  ensemble ttm <ttm_gm> <ttm_gid> <Ce> <rho_e> <kappa_e> <gamma_p> <gamma_s> <v_0> <nx> <ny> <nz> <T_e_init> [{optional_args}]

:attr:`heat_ttm`
^^^^^^^^^^^^^^^^
The full command is::

  ensemble heat_ttm <T> <T_coup> <delta_T> <label_source> <label_sink> <ttm_gm> <ttm_gid> <Ce> <rho_e> <kappa_e> <gamma_p> <gamma_s> <v_0> <nx> <ny> <nz> <T_e_init> [{optional_args}]

For :attr:`heat_ttm`, the first five parameters,
:attr:`<T>`, :attr:`<T_coup>`, :attr:`<delta_T>`, :attr:`<label_source>`, and
:attr:`<label_sink>`, have the same meanings as in :attr:`heat_lan`.

Required parameters
-------------------

For both :attr:`ttm` and :attr:`heat_ttm`:

* :attr:`<ttm_gm>` and :attr:`<ttm_gid>` specify the atom group coupled to the electron grid.
* :attr:`<Ce>` is the electron specific heat per electron.
* :attr:`<rho_e>` is the electron number density.
* :attr:`<kappa_e>` is the electron thermal conductivity in eV/(ps K Å).
* :attr:`<gamma_p>` and :attr:`<gamma_s>` are friction coefficients in amu/ps.
* :attr:`<v_0>` is the threshold velocity in Å/ps.
* :attr:`<nx>`, :attr:`<ny>`, and :attr:`<nz>` are the numbers of electron grid cells in the three directions.
* :attr:`<T_e_init>` is the initial electron temperature in K.

The product :attr:`<Ce> \times <rho_e>` is the volumetric electron heat capacity.

Optional arguments
------------------

:attr:`ttm_out_interval`
^^^^^^^^^^^^^^^^^^^^^^^^
Syntax::

  ttm_out_interval <interval>

Set the output interval for :ref:`ttm_electron_temperature.out <ttm_electron_temperature_out>`.
The default value is 1.

:attr:`ttm_infile`
^^^^^^^^^^^^^^^^^^
Syntax::

  ttm_infile <filename>

Read the initial electron temperature from a file.
The file must contain one line for each cell in the form::

  ix iy iz T_e

The grid indices are 1-based.

:attr:`ttm_properties_file`
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Syntax::

  ttm_properties_file <filename>

Read per-cell electron properties from a file.
The file must contain one line for each cell in the form::

  ix iy iz C_vol kappa_e gamma_p eta

Here, :attr:`C_vol` is the volumetric electron heat capacity,
:attr:`kappa_e` is in eV/(ps K Å), :attr:`gamma_p` is in amu/ps,
and :attr:`eta` is the source absorption efficiency.
This file must define all grid cells and overrides the uniform
:attr:`<Ce>`, :attr:`<rho_e>`, :attr:`<kappa_e>`, and :attr:`<gamma_p>` values.

:attr:`ttm_source`
^^^^^^^^^^^^^^^^^^
Syntax::

  ttm_source <source>

Add a volumetric heat source to the electron grid.
The source strength is in eV/(ps Å\ :math:`^3`).
When :attr:`ttm_properties_file` is used, the source in each cell is multiplied by :attr:`eta`.

:attr:`ttm_active_x`, :attr:`ttm_active_y`, :attr:`ttm_active_z`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Syntax::

  ttm_active_x <range>
  ttm_active_y <range>
  ttm_active_z <range>

Set the active range of the electron grid in each direction.
The range can be :attr:`all`, a single 1-based cell index, or an inclusive interval
such as :attr:`3:10` or :attr:`3-10`.
Cells outside the active region have zero electron temperature and do not exchange
energy with neighboring cells or atoms.

Notes
-----

* The simulation box for the electron grid is the full MD box.
* The electron grid is always uniform in real space.
* Electron-temperature snapshots are written to :ref:`ttm_electron_temperature.out <ttm_electron_temperature_out>`.
* In :attr:`heat_ttm`, the source and sink labels always refer to grouping method 0.

Examples
--------

Uniform pure TTM::

  ensemble ttm 0 0 1.0 1.0 0.005 0.01 0.0 100.0 1 1 12 300

Pure TTM with an input electron-temperature profile and periodic snapshots::

  ensemble ttm 0 0 1.0 1.0 0.005 0.01 0.0 100.0 1 1 12 300 ttm_infile te_init.dat ttm_out_interval 50

Pure TTM with per-cell properties::

  ensemble ttm 0 0 1.0 1.0 0.005 0.01 0.0 100.0 1 1 40 300 ttm_properties_file properties.dat ttm_active_z 6:35 ttm_out_interval 100

Heat source/sink plus TTM::

  ensemble heat_ttm 300 100 30 0 1 1 0 1.0 1.0 0.005 0.02 0.0 100.0 1 1 40 300 ttm_out_interval 100
