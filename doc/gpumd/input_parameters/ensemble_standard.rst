.. _kw_ensemble_standard:

:attr:`ensemble` (standard)
===========================

Select conventional classical MD at fixed energy (NVE), temperature (NVT), or temperature and pressure (NPT).
The :ref:`ensemble theory <ensembles>` describes the thermostat and barostat methods.

Syntax
------

::

    ensemble nve
    ensemble nvt_ber <T_1> <T_2> <T_coup>
    ensemble nvt_nhc <T_1> <T_2> <T_coup>
    ensemble nvt_bdp <T_1> <T_2> <T_coup>
    ensemble nvt_lan <T_1> <T_2> <T_coup>
    ensemble nvt_bao <T_1> <T_2> <T_coup>
    ensemble npt_ber <T_1> <T_2> <T_coup> <pressure_control_parameters>
    ensemble npt_scr <T_1> <T_2> <T_coup> <pressure_control_parameters>

Parameters
----------

``nve`` needs no parameters and applies no thermostat or barostat.
For all NVT and NPT methods, ``T_1`` and ``T_2`` are the initial and final target temperatures in K.
The target changes linearly during the run; use equal values for a constant target.
``T_coup`` is the thermostat coupling time in time steps; see :ref:`units and suggested parameters <choice_of_parameters>`.

The NVT methods use the following thermostats:

* ``nvt_ber``: :ref:`Berendsen <berendsen_thermostat>`.
* ``nvt_nhc``: :ref:`Nosé-Hoover chain <nose_hoover_chain_thermostat>`.
* ``nvt_bdp``: :ref:`Bussi-Donadio-Parrinello <bdp_thermostat>`.
* ``nvt_lan``: :ref:`Langevin <langevin_thermostat>` [Bussi2007a]_.
* ``nvt_bao``: Langevin with BAOAB splitting [Leimkuhler2013]_.

``npt_ber`` uses the :ref:`Berendsen barostat <berendsen_barostat>`; ``npt_scr`` uses :ref:`stochastic cell rescaling <stochastic_cell_rescaling>`.
Both take the same pressure-control parameters, with pressures and estimated elastic moduli in GPa and ``p_coup`` in time steps.

.. _standard_pressure_parameters:

Choose one of the following forms for ``pressure_control_parameters``:

* **Isotropic cell changes**

  .. code::

     <p_hydro> <C_hydro> <p_coup>

  The three box lengths change uniformly according to the hydrostatic pressure :attr:`<p_hydro> = (p_xx + p_yy + p_zz)/3`.
  All directions should have periodic boundary conditions.
  The box must be orthogonal.

* **Independent box lengths**

  .. code::

     <p_xx> <p_yy> <p_zz> <C_xx> <C_yy> <C_zz> <p_coup>

  In this case, the simulation box must be orthogonal.
  The three box lengths will be controlled independently according to their respective target pressures.
  Any direction can be either periodic or nonperiodic and pressure control will only be effective in periodic directions.

* **Triclinic cell changes**

  .. code::

     <p_xx> <p_yy> <p_zz> <p_yz> <p_xz> <p_xy> <C_xx> <C_yy> <C_zz> <C_yz> <C_xz> <C_xy> <p_coup>

  The simulation box must be triclinic and all the directions must be periodic.
  All cell components will be controlled independently according to the 6 target pressure components.

  Elastic constants in literature may use a different nomenclature. The correspondence is as follows:

  | :attr:`C_xx=C_xxxx=C_11`, :attr:`C_yy=C_yyyy=C_22`, :attr:`C_zz=C_zzzz=C_33`,
  | :attr:`C_yz=C_yzyz=C_44`, :attr:`C_zx=C_zxzx=C_55`, :attr:`C_xy=C_xyxy=C_66`

  It is sufficient for the elastic constant tensor :attr:`C_ab` to be a (very rough) estimate as long as it is of the right magnitude.
  It is used to convert the coupling constant (or relaxation time, see :ref:`here <choice_of_parameters>`) of the barostat into suitable internal units.
  If an elastic constant component exceeds 2000 GPa, the coupling constant for that component will be zero and that box component will not be changed.

Output
------

None.

Examples
--------

Maintain a target temperature of 300 K using the BDP thermostat::

    ensemble nvt_bdp 300 300 100

Use isotropic Berendsen pressure control at 0 GPa, with an estimated modulus of 100 GPa::

    ensemble npt_ber 300 300 100 0 100 1000

Notes
-----

None.
