.. _kw_ensemble_sib:
.. _kw_nve_sib:
.. _kw_nvt_sib:
.. _kw_npt_sib:
.. index::
   single: nve_sib (keyword in run.in)
   single: nvt_sib (keyword in run.in)
   single: npt_sib (keyword in run.in)
   single: SIB spin integrator

:attr:`ensemble` (SIB)
======================

The semi-implicit B (SIB) integrator propagates the direction of each
``spin:R:3`` vector with the stochastic Landau--Lifshitz--Gilbert equation.
It preserves each vector's initial magnitude and uses the public
``mforce:R:3`` from a Spin potential as
:math:`\boldsymbol H=-\partial E/\partial\boldsymbol S`.

Syntax
------

For spin-only dynamics with a fixed lattice, use::

    ensemble nve_sib [alpha <value>] [gamma <value>] \
             [stemp <-1|value>] [seed <value>]

For fixed-cell dynamics, use::

    ensemble nvt_sib <T_1> <T_2> <T_coup> \
             [lattice on|off] [alpha <value>] \
             [gamma <value>] [stemp <-1|0|value>] [seed <value>]

For pressure-controlled dynamics, use::

    ensemble npt_sib temp <T_1> <T_2> <pressure_terms> \
             [tperiod <tau_temp>] [pperiod <tau_press>] \
             [alpha <value>] [gamma <value>] \
             [stemp <-1|0|value>] [seed <value>]

The MTTK portion of ``npt_sib`` has the same syntax as
:ref:`npt_mttk <kw_ensemble_mttk>`. SIB options must follow all MTTK options.

Examples
--------

Propagate spins deterministically while keeping the lattice fixed::

    ensemble nve_sib alpha 0.1 stemp -1 seed 12345

Propagate only spins at 300 K::

    ensemble nvt_sib 300 300 100 lattice off alpha 0.1 seed 12345

Run coupled spin--lattice dynamics with the physical default gyromagnetic
ratio::

    ensemble nvt_sib 300 300 100 lattice on alpha 0.1

Run isotropic pressure-controlled spin--lattice dynamics::

    ensemble npt_sib temp 300 300 iso 0 0 \
             tperiod 100 pperiod 1000 alpha 0.1 seed 12345

Parameters
----------

``T_1``, ``T_2``
  Initial and final temperatures in K. SIB uses the linearly ramped target
  temperature for its thermal magnetic noise. With lattice dynamics enabled,
  the same target also controls the lattice thermostat.

``T_coup``
  Nose--Hoover-chain coupling period in timesteps for the lattice. It remains
  required by ``nvt_sib`` when ``lattice off`` but then does not alter the
  spin equation.

``lattice on|off``
  Available only for ``nvt_sib`` and defaults to ``on``. ``off`` keeps atomic
  positions and velocities fixed while evaluating the Spin potential twice
  per timestep for the SIB midpoint and endpoint fields.

``alpha <value>``
  Dimensionless Gilbert damping coefficient. It must be finite and
  non-negative. The default is ``0``; thermal noise is then also zero.

``gamma <value>``
  Positive gyromagnetic factor :math:`\gamma/\hbar` in
  :math:`(\mathrm{eV\,ps})^{-1}`. The default is
  :math:`2/\hbar=3038.53\ (\mathrm{eV\,ps})^{-1}` in GPUMD's unit system.

``stemp <-1|0|value>``
  Controls the thermal magnetic noise. ``-1`` disables it while retaining
  deterministic damping, ``0`` follows the ensemble target temperature, and
  a positive value fixes the spin-noise temperature in K. The default is
  ``0``.

  For ``nve_sib``, the default is ``-1`` because there is no lattice target
  temperature. A fixed positive spin temperature is accepted, while ``0`` is
  rejected. ``nve_sib`` always keeps lattice positions and velocities fixed,
  matching the DynSpin SIB/NVE separation between spin and lattice dynamics.

``seed <value>``
  Positive integer for counter-based Gaussian noise. The default is
  ``12345``. A given atom, timestep, and seed always receive the same noise.

Algorithm
---------

For a unit direction :math:`\boldsymbol e_n`, SIB forms

.. math::

   \boldsymbol\Omega =
   \frac{\gamma\Delta t}{1+\alpha^2}
   [\boldsymbol H+\alpha(\boldsymbol e\times\boldsymbol H)]
   +\boldsymbol\Omega_\mathrm{th}.

The predictor endpoint is an analytic Cayley rotation of
:math:`\boldsymbol e_n`. GPUMD evaluates the second Spin-potential field at
the unnormalized arithmetic chord midpoint
:math:`(\boldsymbol e_n+\boldsymbol e_\mathrm{pred})/2`, reuses the same
thermal increment, and applies the corrected Cayley rotation from the
original direction. A final force evaluation caches the endpoint energy,
force, virial, and magnetic force for output and the next timestep. Thus SIB
uses two Spin-potential evaluations per timestep after the initial force.

The angular-noise standard deviation for a spin magnitude :math:`\mu_s`
follows from matching its variance to the damping mobility
:math:`\gamma\alpha/(1+\alpha^2)` (fluctuation--dissipation):

.. math::

   \sigma_\Omega =
   \sqrt{\frac{2\alpha\gamma k_B T\Delta t}{\mu_s(1+\alpha^2)}}.

Restrictions
------------

SIB requires one Spin potential, ``spin:R:3`` on every atom, a positive
timestep, positive target temperatures, ``alpha >= 0``, ``gamma > 0``, and a
positive seed. Target temperatures are only required by the NVT/NPT forms. A
zero spin vector remains zero. ``nve_sib`` always fixes the lattice and does
not accept ``lattice``; ``npt_sib`` always includes
lattice and cell dynamics and therefore does not accept ``lattice off``.
SIB propagates every nonzero ``spin:R:3`` vector; ``spin_dof_type`` masks the
potential response but is not an integration constraint. In particular, an
excluded type has zero deterministic magnetic field but can still rotate
under thermal noise.

For variable-magnitude dynamics use the independent
:ref:`GLSD ensembles <kw_ensemble_glsd>`. SIB does not accept ``lsf``, ``damping`` or
``mobility``; its fixed-magnitude equation and Cayley integration are unchanged.
