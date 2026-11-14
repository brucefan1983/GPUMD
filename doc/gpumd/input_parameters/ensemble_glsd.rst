.. _kw_ensemble_glsd:
.. _kw_nve_glsd:
.. _kw_nvt_glsd:
.. _kw_npt_glsd:
.. index::
   single: nve_glsd (keyword in run.in)
   single: nvt_glsd (keyword in run.in)
   single: npt_glsd (keyword in run.in)
   single: generalized Langevin spin dynamics

:attr:`ensemble` (GLSD)
=======================

GLSD evolves both direction and magnitude of ``spin:R:3`` magnetic moments.
It implements the generalized Langevin equation of Ma and Dudarev,
Phys. Rev. B **86**, 054416 (2012), Eqs. (18)--(19) and (33), with a stochastic
Heun discretization. It requires a NEP-spin potential.
For fixed-magnitude dynamics use :ref:`SIB <kw_ensemble_sib>`.

Syntax
------

::

    ensemble nve_glsd damping <value> [gamma <value>] [stemp <-1|value>] [seed <value>]
    ensemble nvt_glsd <T_1> <T_2> <T_coup> [lattice on|off] damping <value> [gamma <value>] [stemp <-1|0|value>] [seed <value>]
    ensemble npt_glsd temp <T_1> <T_2> <pressure_terms> [tperiod <tau_temp>] [pperiod <tau_press>] damping <value> [gamma <value>] [stemp <-1|0|value>] [seed <value>]

Parameters
----------

``damping``
  Required, finite, non-negative and dimensionless. It is
  :math:`\eta=\hbar\gamma'_s`, where :math:`\gamma'_s` is the paper's
  Cartesian GLSD dissipation coefficient. It differs from the paper's
  :math:`\gamma_s` and from Gilbert alpha. There is no material-independent
  physical default. Zero disables damping and noise. The corresponding constant mobility is
  :math:`\lambda=\eta g_L^2/\hbar=\eta g^2\hbar`, in muB^2/(eV ps).

``gamma``
  Positive, finite numerical precession coefficient :math:`g=g_L/\hbar`;
  default :math:`2/\hbar \simeq 3038.54` in the eV/ps convention.
  GLSD interprets an explicit value as :math:`g_L=g\hbar`, so changing
  ``gamma`` at fixed ``damping`` also changes :math:`\lambda` quadratically
  and the noise amplitude accordingly. Keyword order has no effect.
  Keep ``gamma`` at its default unless a different precession coefficient is
  intended. This is not the paper's damping parameter.

``stemp``
  ``-1`` disables noise while retaining deterministic damping; ``0`` follows
  the linearly ramped ensemble target; a positive value fixes the spin bath
  temperature in K. Default: ``-1`` for ``nve_glsd``, ``0`` otherwise.
  ``nve_glsd`` rejects ``stemp 0`` because it has no ensemble temperature.

``seed``
  Positive integer; default ``12345``. Keys the counter-based Gaussian stream.

``lattice``
  Only for ``nvt_glsd``, defaults to ``on``. ``off`` fixes positions and
  velocities. ``nve_glsd`` always fixes them; ``npt_glsd`` always evolves
  atoms and the cell. The name ``nve_glsd`` denotes the fixed-lattice driver:
  with positive mobility the spin subsystem dissipates energy, so its energy
  is not conserved even when noise is disabled.

The NVT temperature and lattice coupling parameters have the same meaning as
:ref:`NHC <kw_ensemble_standard>`. ``T_coup`` is required even with
``lattice off``, when it has no effect on spins. The NPT prefix has the syntax
of :ref:`MTTK <kw_ensemble_mttk>`; all GLSD options follow the MTTK options.
GLSD uses dimensionless ``damping``; SIB uses Gilbert ``alpha``.
These keywords select different dissipative equations and are not interchangeable.

Choosing damping
^^^^^^^^^^^^^^^^

For equilibrium sampling, ``damping 0.01`` is a starting point for convergence tests, not a fitted material parameter.
Compare several positive values and smaller time steps, checking energy, magnetization, and moment-length statistics after equilibration.
At a common constant bath temperature, positive constant mobilities have the same continuum Gibbs distribution if it is normalizable and the dynamics equilibrates.
They change relaxation and correlation times. With different bath temperatures or driven fields, steady observables can also depend on mobility.

For physical relaxation times, calibrate the coefficient for the material, state, and bath being studied.
The potential defines energy and forces; it does not determine bath coupling.
For a local radial harmonic mode with curvature :math:`\kappa` and relaxation time :math:`\tau_\parallel`,

.. math::

   \tau_\parallel=\frac{1}{\lambda\kappa}.

This is a local estimate, not a general fit to coupled finite-temperature dynamics.
One scalar mobility controls both longitudinal and transverse relaxation, so a Gilbert damping value alone does not specify this model.

Output
------

None.

Examples
--------

The following examples use dimensionless ``damping 0.01`` as a starting
setting for pilot runs, not a default or a calibrated material parameter.
After loading a NEP-spin potential and a ``model.xyz`` with magnetic moments::

    # Deterministic dissipative spin dynamics; fixed atomic positions
    ensemble nve_glsd damping 0.01 stemp -1
    time_step 0.1
    dump_xyz 100 spin.xyz precision double spin mforce
    dump_restart 1000
    run 1000

Other ensemble choices::

    # Fixed lattice; spins coupled to a 300 K bath
    ensemble nvt_glsd 300 300 100 lattice off damping 0.01 seed 731

    # Coupled spin and lattice dynamics (lattice NHC)
    ensemble nvt_glsd 300 300 100 damping 0.01 seed 731

    # Coupled spin, lattice and cell dynamics (MTTK)
    ensemble npt_glsd temp 300 300 iso 0 0 tperiod 100 pperiod 1000 damping 0.01 seed 731

The example damping is not a fitted Fe parameter, and the timestep is not a
stability guarantee. Check timestep convergence and the potential's coverage
of the sampled moment magnitudes. A potential trained at one fixed magnitude
need not describe longitudinal energetics.

Notes
-----

Use ``dump_xyz ... precision double spin mforce``. GLSD has no independent
``spin_vel`` state; noisy paths have no instantaneous differentiable velocity.
Entering GLSD clears inherited spin velocity and marks it uninitialized;
requesting ``spin_velocity`` output then fails explicitly. Switching later to
TSPIN lets that integrator initialize its own velocity. No spin kinetic energy
is included in the lattice temperature.

For continuation, follow :ref:`Spin restart <spin_restart>`.
A new run initializes its thermostat, barostat, and noise sequence; reusing a seed restarts that sequence.

The same seed and atom order give the same random increments across GPU counts, but floating-point reductions can differ.
Assess stochastic results through converged statistics rather than long-trajectory identity.
Only elements listed in ``spin_type`` evolve and receive Cartesian thermal
noise, including active atoms whose current moment is zero. Other elements
retain zero effective moments.

Method
^^^^^^

The convention follows `Ma and Dudarev (2012) <https://doi.org/10.1103/PhysRevB.86.054416>`_, with moments expressed in units of :math:`\mu_B`.
The dimensionless input is :math:`\eta=\hbar\gamma'_s`, where :math:`\gamma'_s` is the paper's Cartesian dissipation coefficient.
For ``gamma`` :math:`g=g_L/\hbar`, the mobility is :math:`\lambda=\eta g^2\hbar` in :math:`\mu_B^2/(\mathrm{eV\,ps})`.
Startup output reports both damping and mobility.

Let :math:`\mathbf S` denote the numerical magnetic moment in units of
:math:`\mu_B`, :math:`\mathbf H=-\partial E/\partial\mathbf S` the public
magnetic force, :math:`g` the ``gamma`` numerical coefficient and
:math:`\lambda` the constant mobility derived from ``damping``. Time is expressed in ps:

.. math::

   d\mathbf S_i = \mathbf a_i\,dt + \sqrt{2\lambda k_BT}\,d\mathbf W_i,
   \qquad \mathbf a_i=-g\,\mathbf S_i\times\mathbf H_i+\lambda\mathbf H_i.

The precession sign agrees with SIB. There is no Gilbert
:math:`1/(1+\alpha^2)` factor in this mode. With dimensional magnetic moments,
the precession coefficient has units :math:`\mu_B/(\mathrm{eV\,ps})`;
the ``gamma`` input absorbs the numerical Bohr-magneton unit
and is printed in :math:`(\mathrm{eV\,ps})^{-1}`. ``add_mfield`` takes
tesla and performs the Zeeman conversion before integration.

For :math:`r=|\mathbf S|>0`, the deterministic longitudinal and transverse
parts are

.. math::

   \dot r=\lambda\,\mathbf e\cdot\mathbf H,\qquad
   \dot{\mathbf S}_\perp=-g\,\mathbf S\times\mathbf H
      -\frac{\lambda}{r^2}\mathbf S\times(\mathbf S\times\mathbf H),
   \quad \mathbf e=\mathbf S/r.

Both transverse cross-product relations are retained. The transverse direction
damping coefficient is :math:`\lambda/r`, rather than the fixed-magnitude
Gilbert value :math:`g\alpha/(1+\alpha^2)`. One isotropic Cartesian mobility
controls longitudinal and transverse relaxation. Independent rates and
temperature-dependent LLB coefficients are not implemented. The code evaluates
:math:`\lambda\mathbf H` directly, without divisions by :math:`r`. At zero
moment a nonzero magnetic force or thermal noise can create a moment; zero
force at zero temperature leaves it zero.

For fixed positions and a time-independent conservative energy, the continuum
zero-temperature equation satisfies

.. math::

   \frac{dE}{dt}=-\lambda\sum_i |\mathbf H_i|^2\leq0.

Precession does no work. This does not imply strict energy decrease at every
finite numerical step, or for driven fields, a thermostat or moving atoms.

Each step uses one Gaussian vector :math:`\boldsymbol\xi_i` with covariance
:math:`2\lambda k_BT\Delta t\,I`:

.. math::

   \mathbf S^*=\mathbf S_n+\Delta t\,\mathbf a_n+\boldsymbol\xi,\qquad
   \mathbf S_{n+1}=\mathbf S_n+
       \frac{\Delta t}{2}(\mathbf a_n+\mathbf a^*)+\boldsymbol\xi.

The stochastic Heun method is explicit and conditionally stable.
For a radial harmonic well, :math:`\lambda\kappa\Delta t<2` is a stability condition, but accurate results generally require smaller steps.
Check time-step convergence. Unlike SIB, Heun does not preserve spin length exactly during undamped precession.

Noise, equilibrium measure and limits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Independent Cartesian Wiener increments obey
:math:`\langle dW_{ia}dW_{jb}\rangle=\delta_{ij}\delta_{ab}dt`.
The diffusion coefficient is :math:`D=\lambda k_BT`, with
:math:`k_B=8.617343\,10^{-5}` eV/K. Noise is additive: Ito and Stratonovich
interpretations coincide in Cartesian coordinates. The same increment is
reused in predictor and corrector, never redrawn or counted twice.

For a smooth, confining, time-independent energy at constant temperature,
the continuum invariant measure is

.. math::

   P(\{\mathbf S\})\prod_i d^3S_i
     \propto \exp[-E/(k_BT)]\prod_i d^3S_i.

The dissipative probability current vanishes for this density; precession
is divergence-free and tangent to constant energy. In radial variables the
volume element contains :math:`r^2dr\,d\Omega`, and the Ito radial drift
contains :math:`2\lambda k_BT/r`. No extra radial entropy force is added in
Cartesian coordinates. A flat :math:`dr\,d\Omega` measure would be a different
physical model. This is a classical bath, not an automatic quantum or
material-specific longitudinal entropy correction. Finite-step sampling has
bias and is not an exact Boltzmann sampler. Thermalization requires positive
mobility and temperature; ``stemp -1`` has deterministic damping only.

A bounded neural-network energy or pure Zeeman energy need not define a
normalizable distribution over unbounded moments. Short trajectories with
such a model establish only local stability, not global thermal equilibrium
or extrapolation validity. Different spin measures or material-specific
kinetics require separate specification and validation.
