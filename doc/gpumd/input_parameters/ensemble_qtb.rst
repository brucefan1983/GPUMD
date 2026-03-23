.. _kw_ensemble_qtb:
.. index::
   single: nvt_qtb (keyword in run.in)
   single: npt_qtb (keyword in run.in)
   single: Quantum Thermal Bath

:attr:`ensemble` (QTB)
======================

The variants of the :attr:`ensemble` keyword described on this page implement the Quantum Thermal Bath (QTB) method [Dammak2009]_.
The QTB thermostat is a Langevin-type thermostat that uses colored noise with a quantum Bose-Einstein energy spectrum instead of classical white noise.
This allows approximate inclusion of nuclear quantum effects (zero-point energy and quantum heat capacity) in otherwise classical MD simulations.

The :attr:`npt_qtb` variant combines the QTB thermostat with the Parrinello-Rahman (MTTK) barostat [Martyna1994]_, equivalent to LAMMPS ``fix nph`` + ``fix qtb``.


Syntax
------

:attr:`nvt_qtb`
^^^^^^^^^^^^^^^

Run an NVT simulation with the QTB thermostat::

    ensemble nvt_qtb <T_1> <T_2> <T_coup> [f_max <value>] [N_f <value>] [seed <value>]

* :attr:`<T_1>` and :attr:`<T_2>`: Initial and final target temperature (K). The target temperature varies linearly during the run.
* :attr:`<T_coup>`: Thermostat coupling parameter (in units of timestep). Controls the friction coefficient: :math:`\gamma = 1 / (\text{T\_coup} \times dt)`.
* :attr:`f_max`: (Optional, default 200) Maximum frequency of the QTB filter in ps\ :sup:`-1`. Should be larger than the highest phonon frequency in the system.
* :attr:`N_f`: (Optional, default 100) Number of frequency points in the filter. The filter uses :math:`2 N_f` points total.
* :attr:`seed`: (Optional, default 880302) Random number seed for the colored noise generator.

:attr:`npt_qtb`
^^^^^^^^^^^^^^^

Run an NPT simulation with the QTB thermostat and Parrinello-Rahman (MTTK) barostat::

    ensemble npt_qtb <direction> <p_1> <p_2> temp <T_1> <T_2> tperiod <tau_T> pperiod <tau_p> [f_max <value>] [N_f <value>] [seed <value>]

Pressure control parameters:

* :attr:`<direction>`: One or more of ``iso``, ``aniso``, ``tri``, ``x``, ``y``, ``z``. Same syntax as :ref:`npt_mttk <mttk>`.
* :attr:`<p_1>` and :attr:`<p_2>`: Initial and final target pressure (GPa).

Temperature and coupling parameters:

* :attr:`temp <T_1> <T_2>`: Initial and final target temperature (K).
* :attr:`tperiod <tau_T>`: QTB thermostat coupling period (in units of timestep). Controls friction: :math:`\gamma = 1 / (\text{tperiod} \times dt)`.
* :attr:`pperiod <tau_p>`: Barostat coupling period (in units of timestep, must be :math:`\geq 200`).

QTB-specific optional parameters (same as :attr:`nvt_qtb`):

* :attr:`f_max`: Maximum frequency (ps\ :sup:`-1`, default 200).
* :attr:`N_f`: Number of frequency points (default 100).
* :attr:`seed`: Random seed (default 880302).


Examples
--------

NVT-QTB
^^^^^^^^

.. code-block:: rst

    ensemble nvt_qtb 300 300 100

Run at 300 K with QTB thermostat. The coupling parameter is 100 timesteps.

.. code-block:: rst

    ensemble nvt_qtb 300 300 100 f_max 150 N_f 200

Same as above but with custom filter parameters.

NPT-QTB
^^^^^^^^

.. code-block:: rst

    ensemble npt_qtb iso 0 0 temp 300 300 tperiod 100 pperiod 1000

Run at 300 K and 0 GPa with isotropic pressure control.

.. code-block:: rst

    ensemble npt_qtb aniso 0 0 temp 300 300 tperiod 100 pperiod 1000 f_max 200 N_f 100 seed 12345

Anisotropic pressure control with explicit QTB parameters.

.. code-block:: rst

    ensemble npt_qtb x 5 5 y 0 0 z 0 0 temp 300 300 tperiod 100 pperiod 1000

Apply 5 GPa along x and 0 GPa along y and z.


Notes
-----

* The QTB method generates colored noise whose power spectrum matches the quantum energy distribution :math:`E(\omega) = \hbar\omega[\frac{1}{2} + n_{BE}(\omega, T)]`, where :math:`n_{BE}` is the Bose-Einstein distribution.
* The kinetic temperature reported in ``thermo.out`` will be higher than the target temperature due to zero-point energy contributions. This is expected behavior.
* For liquid water at 300 K, the kinetic temperature is typically around 1000-1100 K.
* The :attr:`npt_qtb` ensemble uses the MTTK (Martyna-Tuckerman-Tobias-Klein) integrator for pressure control, which is the same as :ref:`npt_mttk <mttk>` but with the Nosé-Hoover chain thermostat replaced by the QTB thermostat.
* The :attr:`f_max` parameter should be set larger than the highest phonon frequency in the system. For water, 200 ps\ :sup:`-1` is sufficient.


References
----------

.. [Dammak2009] H. Dammak, Y. Chalopin, M. Laroche, M. Hayoun, and J.-J. Greffet, *Quantum Thermal Bath for Molecular Dynamics Simulation*, Phys. Rev. Lett. **103**, 190601 (2009).