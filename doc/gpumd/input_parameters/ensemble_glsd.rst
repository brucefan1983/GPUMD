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
Heun discretization. It requires a Spin potential and uses its existing
magnetic force, atomic force and virial interfaces.
For fixed-magnitude dynamics use :ref:`SIB <kw_ensemble_sib>`.

Quick start
-----------

The following examples use dimensionless ``damping 0.01`` as a starting
setting for pilot runs, not a default or a calibrated material parameter.
No equilibrium-moment scan or element table is required to set damping.
Read the guidance below before interpreting physical relaxation times. After loading a Spin potential and a
``model.xyz`` with magnetic moments::

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

Syntax and parameters
---------------------

::

    ensemble nve_glsd damping <value> [gamma <value>] [stemp <-1|value>] [seed <value>]
    ensemble nvt_glsd <T_1> <T_2> <T_coup> [lattice on|off] \
             damping <value> [gamma <value>] [stemp <-1|0|value>] [seed <value>]
    ensemble npt_glsd temp <T_1> <T_2> <pressure_terms> \
             [tperiod <tau_temp>] [pperiod <tau_press>] \
             damping <value> [gamma <value>] [stemp <-1|0|value>] [seed <value>]

``damping``
  Required, finite, non-negative and dimensionless. It is
  :math:`\eta=\hbar\gamma'_s`, where :math:`\gamma'_s` is the paper's
  Cartesian GLSD dissipation coefficient. This is a new input normalization
  of that coefficient, not another physical term and not the paper's
  :math:`\gamma_s` or Gilbert alpha. There is no material-independent
  physical default. Zero disables damping and noise. The code converts it
  once per ensemble specification to the constant internal mobility
  :math:`\lambda=\eta g_L^2/\hbar=\eta g^2\hbar`, in muB^2/(eV ps).
  It is never inferred from element identity, initial or instantaneous
  magnetic moments, volume or force curvature.

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
GLSD rejects ``alpha``, ``lsf`` and the earlier unreleased ``mobility``
keyword. SIB rejects ``damping``, ``mobility`` and ``lsf``. Selecting the
ensemble name selects the physical model. Do not rename an old mobility
value without converting it.

Choosing damping
----------------

For equilibrium pilot runs, ``damping 0.01`` is a convenient numerical
starting choice, not a demonstrated universal optimum or a material fit.
Compare, for example, ``0.005``, ``0.01`` and ``0.02`` after equilibration;
check correlation times and shrink the timestep before comparing statistical
averages. This suggested bracket is not a calibrated range or a confidence
interval. More damping is not guaranteed to improve mixing or stability.
For quantitative time-dependent predictions, use independently calibrated
kinetics instead of this pilot-run choice.

The transformation removes units from the input; it does not remove physical
material dependence. A common damping across elements defines a common
model bath coupling, not identical relaxation times for all elements.

A Spin potential specifies energy and forces; these do not by themselves
specify coupling to the dissipative bath. The fluctuation--dissipation relation
fixes the noise amplitude **given** damping, gamma and temperature; it
does not determine damping. Merely obtaining a stable run or the correct
equilibrium temperature cannot identify a physical relaxation rate.

For equilibrium sampling
  At a common, constant bath temperature, positive constant mobilities have
  the same continuum Cartesian Gibbs invariant distribution, provided it is
  normalizable and the dynamics equilibrates. Mobility changes equilibration
  and correlation times. Choose it by pilot runs: check stationarity and
  mixing of energy, magnetization and moment-length statistics; repeat at
  smaller timestep and several positive mobilities. Compare equilibrium
  observables within sampling uncertainty. Do not infer physical relaxation
  times from an uncalibrated sampling coefficient. With unequal spin/lattice
  bath temperatures or driven fields, even steady observables can depend on
  mobility; the equilibrium argument does not apply.

For physical time-dependent dynamics
  Calibrate against a relaxation measurement or an independently justified
  dynamical calculation for the same material, temperature, state and bath.
  A coefficient from another Hamiltonian or a particular Fe experiment is
  only a candidate, not a transferable material constant. Report its source
  and test sensitivity. Do not increase it just to accelerate the simulation.

For an initial local estimate
  In a fixed-environment radial mode near a stable nonzero moment,
  :math:`E(r)\simeq E(r_0)+\kappa(r-r_0)^2/2`, with positive curvature
  :math:`\kappa` in eV/muB^2, the linear deterministic radial equation gives

  .. math::

     \frac{d\, \delta r}{dt}=-\lambda\kappa\, \delta r,\qquad
     \tau_\parallel=\frac{1}{\lambda\kappa},\qquad
     \lambda=\frac{1}{\kappa\tau_\parallel}.

  This is a local estimate derived from the implemented equation, not a
  general fit to a coupled finite-temperature relaxation spectrum. Curvature
  alone is insufficient: a target relaxation time is also needed. For example,
  hypothetical :math:`\kappa=0.4` eV/muB^2 and
  :math:`\tau_\parallel=0.5` ps give mobility ``5``. These numbers are an
  illustration, not Fe data. A sampling-only chosen time is a numerical
  choice, not a measured material timescale.

Knowing a Gilbert damping alpha alone does not uniquely calibrate this full
variable-magnitude model. One scalar mobility constrains both transverse and
longitudinal relaxation; if the measured rates cannot be matched together,
this isotropic GLSD model is insufficient. Relabeling alpha or adding a default
would not resolve that physical limitation.

For each candidate, check timestep convergence separately from parameter
sensitivity. In the simple radial harmonic model the explicit-Heun stability
condition is :math:`\lambda\kappa\Delta t<2` (time in ps), but accuracy needs
smaller steps and precession or other coupled modes may impose tighter limits.
Without kinetic data, users can study equilibrated static statistics under the
stated assumptions, or report a mobility sensitivity study; the potential
alone cannot justify quantitative relaxation times.

Paper-to-code conventions
-------------------------

The reference is `Ma and Dudarev (2012)
<https://doi.org/10.1103/PhysRevB.86.054416>`_, specifically Eqs. (18)--(19)
for drift, Eqs. (20)--(22) for transverse projection, and Eq. (33) for noise.
No LAMMPS implementation is used as the specification.

Write the paper's dimensionless angular-momentum vector as :math:`\mathbf s`
and its Landé factor as :math:`g_L`. The paper defines the magnetic moment
:math:`\mathbf M=-g_L\mu_B\mathbf s`. GPUMD stores the numerical magnetic
moment :math:`\mathbf S=\mathbf M/\mu_B=-g_L\mathbf s`. Hence
:math:`\mathbf H_s=-g_L\mathbf H`, where
:math:`\mathbf H=-\partial E/\partial\mathbf S`. The paper's equation is

.. math::

   d\mathbf s = [\hbar^{-1}\mathbf s\times\mathbf H_s
     +\gamma'_s\mathbf H_s]dt +\sqrt{2\gamma'_s k_BT}\,d\mathbf W_s.

Changing variables gives the equation below with numerical coefficients
:math:`g=g_L/\hbar` (input ``gamma``) and
:math:`\lambda=g_L^2\gamma'_s` (internal mobility), when time is in ps.
The user input is :math:`\eta=\hbar\gamma'_s` (``damping``).
For dimensional moments the latter is
:math:`\lambda_M=(g_L\mu_B)^2\gamma'_s`.
The Wiener process changes sign, which does not change its distribution.
Thus the negative precession cross product in code is the paper's positive
cross product expressed in magnetic-moment coordinates, not a sign change
in the physical model. The default uses :math:`g_L=2`.
The numerical Fe example needs care. Section III.C quotes
:math:`\gamma'_s=5.88\times10^{13}` eV^-1 s^-1, which literally maps to
mobility ``235.2`` at :math:`g_L=2`. However, Eq. (28) states
:math:`\gamma'_s=s_0^2\gamma_s/\hbar`. Combining the paper's
:math:`M_0=2.2\mu_B`, :math:`s_0=M_0/(g_L\mu_B)=1.1`, and
:math:`\gamma_s=0.008` gives :math:`\gamma'_s=1.47065\times10^{13}`
eV^-1 s^-1 and mobility ``58.826``. These printed definitions and numbers
leave a factor-of-four normalization discrepancy. Do not treat the quoted
``235.2`` as an unambiguous experimental calibration. The validation case
using it checks coordinate and time-unit conversion for that specified
coefficient only. In the dimensionless convention the two routes give
approximately 0.03870 and 0.00968, respectively; neither equals 0.008.

The damping ``0.008`` was obtained by fitting laser-driven Fe-film
magnetization in the coupled spin--lattice--electron model of
`Ma, Dudarev and Woo (2012), Sec. III.B
<https://doi.org/10.1103/PhysRevB.85.184301>`_. That fit also involved
lattice--electron coupling and absorbed pulse energy. Transferring it to a
new potential is an assumption, not a new experimental fit. If this prior
is adopted explicitly, matching transverse damping at a chosen reference
moment :math:`m_0=|S_0|` gives :math:`\lambda=m_0^2\gamma_s/\hbar`
and :math:`\eta=(m_0/g_L)^2\gamma_s`.
Keep the resulting mobility constant; recomputing it from the instantaneous
moment would change the model and require a different noise treatment.

Heun is the numerical discretization chosen here, not a claim to reproduce
the paper's particular numerical implementation. The paper's illustrative
Heisenberg--Landau Hamiltonian, fitted Fe coefficients and temperature
estimator involving energy Hessians are not added to GPUMD: the supplied
Spin potential provides the Hamiltonian. Standard thermo temperature remains
the lattice kinetic temperature, not the paper's Eq. (34) spin temperature.

Converting an earlier input
---------------------------

For the earlier unreleased ``mobility L`` interface, retain the same gamma
:math:`g` and replace it by ``damping`` equal to
:math:`\eta=L/(g^2\hbar)`. At default gamma this is :math:`L\hbar/4`.
GPUMD retains its existing rounded constant
:math:`\hbar=0.06465412\times10.18051/1000
=0.0006582119152012` eV ps, so the default-gamma conversions are:

.. list-table::
   :header-rows: 1

   * - Earlier mobility
     - New damping (approximately)
   * - 10
     - 0.001645529788003
   * - 60.7707
     - 0.01
   * - 77.8486
     - 0.0128102

Use the full conversion expression when matching an old trajectory; rounded
examples need not reproduce it exactly. A non-default gamma requires the
full formula. Startup output reports dimensionless damping and converted
mobility. Existing GLSD restart files keep their counter and seed; continue
with the equivalent damping, the same gamma and other settings. The file
format is unchanged.

Equation and integration
------------------------

Let :math:`\mathbf S` denote the numerical magnetic moment in units of
:math:`\mu_B`, :math:`\mathbf H=-\partial E/\partial\mathbf S` the public
magnetic force, :math:`g` the existing ``gamma`` numerical coefficient and
:math:`\lambda` the constant internal mobility derived from ``damping``. Time is converted from fs to ps:

.. math::

   d\mathbf S_i = \mathbf a_i\,dt + \sqrt{2\lambda k_BT}\,d\mathbf W_i,
   \qquad \mathbf a_i=-g\,\mathbf S_i\times\mathbf H_i+\lambda\mathbf H_i.

The precession sign agrees with SIB. There is no Gilbert
:math:`1/(1+\alpha^2)` factor in this mode. With dimensional magnetic moments,
the precession coefficient has units :math:`\mu_B/(\mathrm{eV\,ps})`;
GPUMD's historical ``gamma`` input absorbs the numerical Bohr-magneton unit
and is printed in :math:`(\mathrm{eV\,ps})^{-1}`. ``add_mfield`` still takes
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

The first force is cached at the starting endpoint. After the lattice first
half-step and spin predictor, the potential and external field are recomputed
at the predictor. The lattice second half-step and spin corrector follow,
then a full endpoint evaluation refreshes energy, force, virial and magnetic
force. Stateful mechanical modifiers are sampled once, as in SIB. There are
two potential evaluations per step after initialization. Heun is second order
for smooth deterministic drift, and strong order one for additive noise. It
is explicit and conditionally stable. A radial harmonic well requires
:math:`\lambda k\Delta t<2`; accuracy generally needs a much smaller step.
Undamped precession has finite-step norm drift with Heun, unlike Cayley SIB.
Use timestep convergence and SIB when exact fixed-length preservation
is required.

Noise, equilibrium measure and limits
-------------------------------------

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

Output, restart and multiple GPUs
---------------------------------

Use ``dump_xyz ... precision double spin mforce``. GLSD has no independent
``spin_vel`` state; noisy paths have no instantaneous differentiable velocity.
Entering GLSD clears inherited spin velocity and marks it uninitialized;
requesting ``spin_velocity`` output then fails explicitly. Switching later to
TSPIN lets that integrator initialize its own velocity. No spin kinetic energy
is included in the lattice temperature.

``dump_restart`` writes full-precision moments plus ``spin_glsd_step`` and
``spin_glsd_seed`` in ``restart.xyz`` for this mode. To continue, copy it to
``model.xyz`` and repeat the ensemble parameters, field, timestep, seed and
atom order. The seed must match. The counter also continues across successive
GLSD ``run`` blocks. Old XYZ files without these fields start at counter zero.
Selecting another integrator clears this GLSD counter. Restart preserves the
noise stream. With lattice motion enabled, ``nvt_glsd`` also saves the NHC
chain masses, positions and momenta; ``npt_glsd`` saves the MTTK thermostat
and barostat. Both retain the original temperature/pressure protocol phase
and global time; see the :ref:`continuation procedure <kw_dump_restart>`.
Repeat the original ensemble endpoints and run the remaining steps. NHC
masses keep their original starting-temperature convention. ``lattice off``
has no atomic thermostat to checkpoint. Other stateful modifiers and analysis
histories are outside this thermostat checkpoint contract.

Integration and Gaussian generation run on GPU 0, keyed by seed, original
atom index, step and component. Multi-GPU NEP distributes predictor/endpoint
spins, computes owner forces with complete halos and gathers owner outputs
into original atom order. It neither generates per-device noise nor sums
ghost magnetic forces. Identical input order and seed therefore give identical
random increments on one and four GPUs. Neighbor ordering and floating-point
reductions can still differ; compare deterministic short-time errors and
stochastic distributions, not long noisy trajectories bitwise.
``spin_dof_type`` is a potential-response mask, not a frozen-spin mask:
Cartesian thermal noise acts on every atom, including zero moments.
