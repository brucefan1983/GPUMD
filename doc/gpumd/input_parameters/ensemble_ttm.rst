.. _kw_ensemble_ttm:

``ensemble`` (TTM)
======================

This page describes the two-temperature-model (TTM) integrators
``ttm`` and ``heat_ttm``.

Syntax
------

``ttm``
^^^^^^^^^^^
The full command is::

  ensemble ttm <grouping_method> <group_id> <Ce> <rho_e> <kappa_e> <gamma_p> <gamma_s> <v_0> <nx> <ny> <nz> <T_e_init> [{optional_args}]

``heat_ttm``
^^^^^^^^^^^^^^^^
The full command is::

  ensemble heat_ttm <T> <T_coup> <delta_T> <label_source> <label_sink> <grouping_method> <group_id> <Ce> <rho_e> <kappa_e> <gamma_p> <gamma_s> <v_0> <nx> <ny> <nz> <T_e_init> [{optional_args}]

Parameters
----------

For ``heat_ttm``, the first five parameters,
``<T>``, ``<T_coup>``, ``<delta_T>``, ``<label_source>``, and
``<label_sink>``, have the same meanings as in ``heat_lan``.

For both ``ttm`` and ``heat_ttm``:

* ``<grouping_method>`` and ``<group_id>`` specify the atom group coupled to the electron grid.
* ``<Ce>`` is the electron specific heat per electron in eV/K.
* ``<rho_e>`` is the electron number density in Å\ :math:`^{-3}`.
* ``<kappa_e>`` is the electron thermal conductivity in eV/(ps K Å).
* ``<gamma_p>`` and ``<gamma_s>`` are friction coefficients in amu/ps.
* ``<v_0>`` is the threshold velocity in Å/ps.
* ``<nx>``, ``<ny>``, and ``<nz>`` are the numbers of electron grid cells in the three directions.
* ``<T_e_init>`` is the initial electron temperature in K.

The product ``<Ce>`` × ``<rho_e>`` is the volumetric electron heat capacity in eV/(K Å\ :math:`^3`).

Optional arguments
^^^^^^^^^^^^^^^^^^

``ttm_out_interval``
~~~~~~~~~~~~~~~~~~~~~~~~
Syntax::

  ttm_out_interval <interval>

Set the output interval for :ref:`ttm_electron_temperature.out <ttm_electron_temperature_out>`.
The default value is 1.

``ttm_infile``
~~~~~~~~~~~~~~~~~~
Syntax::

  ttm_infile <filename>

Read the initial electron temperature from a file.
The file must contain one line for each cell in the form::

  ix iy iz T_e

The grid indices are 1-based.

``ttm_properties_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Syntax::

  ttm_properties_file <filename>

Read per-cell electron properties from a file.
The file must contain one line for each cell in the form::

  ix iy iz C_vol kappa_e gamma_p eta

Here, ``C_vol`` is the volumetric electron heat capacity in eV/(K Å\ :math:`^3`),
``kappa_e`` is in eV/(ps K Å), ``gamma_p`` is in amu/ps,
and ``eta`` is the dimensionless source absorption efficiency.
This file must define all grid cells and overrides the uniform
``<Ce>`` × ``<rho_e>``, ``<kappa_e>``, and ``<gamma_p>`` values.

``ttm_source``
~~~~~~~~~~~~~~~~~~
Syntax::

  ttm_source <source>

Add a volumetric heat source to the electron grid.
The source strength is in eV/(ps Å\ :math:`^3`).
When ``ttm_properties_file`` is used, the source in each cell is multiplied by ``eta``.

``ttm_active_x``, ``ttm_active_y``, ``ttm_active_z``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Syntax::

  ttm_active_x <range>
  ttm_active_y <range>
  ttm_active_z <range>

Set the active range of the electron grid in each direction.
The range can be ``all``, a single 1-based cell index, or an inclusive interval
such as ``3:10`` or ``3-10``.
Cells outside the active region have zero electron temperature and do not exchange
energy with neighboring cells or atoms.

Output
------

Electron-temperature snapshots are written to :ref:`ttm_electron_temperature.out <ttm_electron_temperature_out>`.

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

Notes
-----

* The simulation box for the electron grid is the full MD box.
* The electron grid is always uniform in real space.
* In ``heat_ttm``, the source and sink labels always refer to grouping method 0.
