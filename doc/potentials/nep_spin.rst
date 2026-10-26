.. _nep_spin:
.. index::
   single: Spin potential
   single: NEP_Spin
   single: magnetic force

Spin potential
**************

``NEP_Spin`` extends the :ref:`NEP4 formalism <nep_formalism>` to a local
energy surface that depends on both atomic positions and Cartesian spin
vectors. The total energy is the only learned scalar output. Atomic forces,
virials, magnetic forces, and spin torques are derivatives of this same energy
and are not produced by independent response networks.

The current format uses ``nep``, ``format_version 1``, and ``model nep_spin``
with ``spin_version 1`` and named parameter and scaler blocks. It uses the order/compression (O/C)
descriptor described below. An optional ``zbl`` field adds a spin-independent
universal ZBL term. Flexible ZBL (``zbl 0 0``) is not supported.
``spin_version`` identifies the descriptor and parameter-layout
contract, independently of the file syntax version and ``spin_order``.

Energy model
============

For atom :math:`i`, the ordinary structural descriptor
:math:`\boldsymbol{q}^{i,\mathrm{str}}` is concatenated with a magnetic
descriptor :math:`\boldsymbol{q}^{i,\mathrm{spin}}`:

.. math::

   \boldsymbol{q}^{i}
   = \boldsymbol{q}^{i,\mathrm{str}}
   \mathbin{\|} \boldsymbol{q}^{i,\mathrm{spin}}.

After componentwise scaling, this vector enters the same one-hidden-layer,
element-dependent neural network used by ordinary NEP4. The total energy is

.. math::

   U(\{\boldsymbol{r}_i,\boldsymbol{s}_i\})
   = \sum_i \left[
       U_i(\boldsymbol{q}^{i}) + E^0_{Z_i}
     \right] + U_\mathrm{ZBL},

where :math:`\boldsymbol{r}_i` is the position, :math:`\boldsymbol{s}_i` is a
Cartesian magnetic-moment vector in Bohr magnetons (muB), :math:`Z_i` denotes the atom type, and
:math:`E^0_{Z_i}` is a fitted per-type energy baseline. The ZBL term is present
when ZBL is enabled and contains no spin dependence. The reference energies
use the shared ``energy_reference`` convention.

This architecture allows the nonlinear network to combine structural and
magnetic coordinates. Therefore, ``spin_order`` describes the hierarchy of
primitive magnetic contractions supplied to the network; it is not the body
order of the final energy.

Spin3 radial channels
=====================

Let

.. math::

   \boldsymbol{e}_{ij}
   = \frac{\boldsymbol{r}_j-\boldsymbol{r}_i}{r_{ij}}

be the minimum-image unit vector from center atom :math:`i` to neighbor
:math:`j`. For each compressed radial channel :math:`c=1,\ldots,C`, Spin3
forms the type-dependent edge weight

.. math::

   w^i_{jc}
   = \sum_{k=0}^{8} c^{Z_i Z_j}_{ck} f_k(r_{ij}),

where

.. math::

   f_k(r)
   = \frac{1}{2}\left[
       T_k\left(2(r/r_\mathrm{c}^{\mathrm{spin}}-1)^2-1\right)+1
     \right] f_\mathrm{c}(r)

and

.. math::

   f_\mathrm{c}(r)
   = \begin{cases}
       \frac{1}{2}\left[1+\cos(\pi r/r_\mathrm{c}^{\mathrm{spin}})\right],
       & r\leq r_\mathrm{c}^{\mathrm{spin}},\\
       0, & r>r_\mathrm{c}^{\mathrm{spin}}.
     \end{cases}

Thus :ref:`spin_compress <kw_spin_compress>` controls a shared latent radial
rank rather than selecting individual Hamiltonian terms. All contraction
families use the same :math:`C` radial channels.

Canonical edge-moment bank
==========================

Spin3 organizes each neighbor edge into a canonical collection of scalar,
vector, rank-2, and product-tensor objects. Define the symmetric traceless
product

.. math::

   \mathcal{Q}(\boldsymbol{a},\boldsymbol{b})
   = \frac{1}{2}\left(
       \boldsymbol{a}\otimes\boldsymbol{b}
       +\boldsymbol{b}\otimes\boldsymbol{a}
     \right)
     -\frac{\boldsymbol{a}\cdot\boldsymbol{b}}{3}\boldsymbol{I}.

The implementation stores this tensor in five orthonormal components,

.. math::

   \operatorname{stf}_5(\boldsymbol{a},\boldsymbol{b})
   = \left(
   \frac{2a_zb_z-a_xb_x-a_yb_y}{\sqrt{6}},
   \frac{a_xb_x-a_yb_y}{\sqrt{2}},
   \frac{a_xb_y+a_yb_x}{\sqrt{2}},
   \frac{a_yb_z+a_zb_y}{\sqrt{2}},
   \frac{a_zb_x+a_xb_z}{\sqrt{2}}
   \right).

For each center :math:`i` and channel :math:`c`, the neighbor reduction builds
the following moment bank:

.. math::

   \boldsymbol{\rho}^{s}_{ic}
   &= \sum_j w^i_{jc}\boldsymbol{s}_j,\\
   \boldsymbol{\rho}^{r}_{ic}
   &= \sum_j w^i_{jc}\boldsymbol{e}_{ij},\\
   \rho^{rs}_{ic}
   &= \sum_j w^i_{jc}(\boldsymbol{e}_{ij}\cdot\boldsymbol{s}_j),\\
   \boldsymbol{\rho}^{r\times s}_{ic}
   &= \sum_j w^i_{jc}(\boldsymbol{e}_{ij}\times\boldsymbol{s}_j),\\
   \boldsymbol{\rho}^{\mathrm{stf}(rs)}_{ic}
   &= \sum_j w^i_{jc}\operatorname{stf}_5(
       \boldsymbol{e}_{ij},\boldsymbol{s}_j),\\
   \boldsymbol{\rho}^{\mathrm{stf}(rr)}_{ic}
   &= \sum_j w^i_{jc}\operatorname{stf}_5(
       \boldsymbol{e}_{ij},\boldsymbol{e}_{ij}),\\
   \boldsymbol{\rho}^{\mathrm{stf}(rr)s}_{ic}
   &= \sum_j w^i_{jc}\operatorname{stf}_5(
       \boldsymbol{e}_{ij},\boldsymbol{e}_{ij})\otimes\boldsymbol{s}_j,\\
   \boldsymbol{\rho}^{(s\cdot s)s}_{ic}
   &= \sum_j w^i_{jc}(\boldsymbol{s}_i\cdot\boldsymbol{s}_j)
       \boldsymbol{s}_j.

The last two objects have 15 and 3 components, respectively. Together with
the scalar and vector entries above, each radial channel has a 38-component
canonical moment state. Most density-derived descriptor coordinates are
scalar contractions of this bank; the moments themselves are not passed
directly to the neural network. Direct one-edge sums and same-edge
channel-pair accumulators are reduced separately.

.. _nep_spin_contractions:

O/C contraction hierarchy
==========================

The hierarchy is nested: at fixed :math:`C`, every order-1 coordinate is a
prefix of order 2, and every order-2 coordinate is a prefix of order 3.
Representative exact contractions are listed below.

Order 1 contains local and direct-edge magnetic information such as

.. math::

   q^{i,\mathrm{spin}}_\mathrm{local}
   &= |\boldsymbol{s}_i|^2,\\
   q^{i,\mathrm{spin}}_{c,\mathrm{dot}}
   &= \boldsymbol{s}_i\cdot\boldsymbol{\rho}^{s}_{ic}
    = \sum_j w^i_{jc}(\boldsymbol{s}_i\cdot\boldsymbol{s}_j),\\
   q^{i,\mathrm{spin}}_{c,\mathrm{neighbor}}
   &= \sum_j w^i_{jc}|\boldsymbol{s}_j|^2.

When joint spin--orbit channels with :math:`l_\mathrm{max}=2` are enabled,
order 1 also contains the rank-2 contractions

.. math::

   \sum_j w^i_{jc}\,
   \mathcal{Q}(\boldsymbol{s}_i,\boldsymbol{s}_j)
   :\mathcal{Q}(\boldsymbol{e}_{ij},\boldsymbol{e}_{ij})

and

.. math::

   \mathcal{Q}(\boldsymbol{s}_i,\boldsymbol{s}_i)
   :\boldsymbol{\rho}^{\mathrm{stf}(rr)}_{ic}.

Order 2 adds edge powers, density norms, and correlations between two magnetic
density legs. Examples include

.. math::

   \sum_j w^i_{jc}(\boldsymbol{s}_i\cdot\boldsymbol{s}_j)^2,
   \qquad
   |\boldsymbol{\rho}^{s}_{ic}|^2,
   \qquad
   (\rho^{rs}_{ic})^2,
   \qquad
   |\boldsymbol{\rho}^{r\times s}_{ic}|^2,

and

.. math::

   \boldsymbol{\rho}^{s}_{ic}\cdot
   \boldsymbol{\rho}^{(s\cdot s)s}_{ic}.

For a radial-channel pair :math:`c\leq d`, Spin3 keeps same-edge and
distinct-neighbor correlations separately:

.. math::

   S^i_{cd}
   &= \sum_j w^i_{jc}w^i_{jd}
      (\boldsymbol{s}_i\cdot\boldsymbol{s}_j)^2,\\
   D^i_{cd}
   &= (\boldsymbol{s}_i\cdot\boldsymbol{\rho}^{s}_{ic})
      (\boldsymbol{s}_i\cdot\boldsymbol{\rho}^{s}_{id})-S^i_{cd}\\
   &= \sum_{j\neq k}w^i_{jc}w^i_{kd}
      (\boldsymbol{s}_i\cdot\boldsymbol{s}_j)
      (\boldsymbol{s}_i\cdot\boldsymbol{s}_k).

This separation prevents a two-density contraction from conflating repeated
use of one neighbor edge with a correlation between distinct neighbors.

Spin3 additionally resolves the distinct-neighbor exchange by spatial angular
rank. Define

.. math::

   \boldsymbol{A}^{(1)}_{ic}
   &=\sum_j w^i_{jc}(\boldsymbol{s}_i\cdot\boldsymbol{s}_j)
     \boldsymbol{e}_{ij},\\
   \boldsymbol{A}^{(2)}_{ic}
   &=\sum_j w^i_{jc}(\boldsymbol{s}_i\cdot\boldsymbol{s}_j)
     \operatorname{stf}_5(\boldsymbol{e}_{ij},\boldsymbol{e}_{ij}).

For every :math:`c\leq d`, the new coordinates are

.. math::

   D^{i,(1)}_{cd}
   &=\boldsymbol{A}^{(1)}_{ic}\cdot\boldsymbol{A}^{(1)}_{id}-S^i_{cd},\\
   D^{i,(2)}_{cd}
   &=\frac{3}{2}\boldsymbol{A}^{(2)}_{ic}\cdot
     \boldsymbol{A}^{(2)}_{id}-S^i_{cd}.

The subtraction removes the repeated-edge contribution, while the
:math:`3/2` factor compensates the norm of the rank-2 STF representation.
These blocks distinguish angular exchange environments that have the same
scalar distinct-neighbor contraction.

Joint spin--orbit contractions use four learned radial-leg mixing matrices
:math:`A^{(\alpha)}\in\mathbb{R}^{C\times C}`. For any moment family
:math:`\boldsymbol{\rho}`, a projected leg is

.. math::

   \overline{\boldsymbol{\rho}}^{(\alpha)}_{ic}
   = \sum_{d=1}^{C} A^{(\alpha)}_{cd}\boldsymbol{\rho}_{id}.

This permits non-collinear contraction legs while retaining one shared radial
seam. Representative order-2 joint invariants are

.. math::

   \left(
     \overline{\boldsymbol{\rho}}^{r,(0)}_{ic}
     \times
     \overline{\boldsymbol{\rho}}^{r,(1)}_{ic}
   \right)\cdot
   \left(
     \boldsymbol{s}_i\times
     \overline{\boldsymbol{\rho}}^{s,(0)}_{ic}
   \right)

and

.. math::

   \operatorname{axial}\left([
     \overline{\mathcal{Q}}^{(0)}_{ic},
     \overline{\mathcal{Q}}^{(1)}_{ic}
   ]\right)\cdot
   \left(
     \boldsymbol{s}_i\times
     \overline{\boldsymbol{\rho}}^{s,(0)}_{ic}
   \right),

where :math:`[\mathcal{A},\mathcal{B}]=\mathcal{A}\mathcal{B}-
\mathcal{B}\mathcal{A}` and ``axial`` maps the resulting antisymmetric tensor
to an axial vector.

Order 3 adds dot-gated and triple-density contractions. Its direct-edge member
is

.. math::

   \sum_j w^i_{jc}
   (\boldsymbol{s}_i\cdot\boldsymbol{s}_j)
   \left(|\boldsymbol{s}_i|^2+|\boldsymbol{s}_j|^2\right),

while representative joint contractions include scalar triple products such
as

.. math::

   \overline{\boldsymbol{\rho}}^{r,(0)}_{ic}\cdot
   \left(
     \overline{\boldsymbol{\rho}}^{s,(1)}_{ic}
     \times
     \overline{\boldsymbol{\rho}}^{r\times s,(2)}_{ic}
   \right).

These coordinates supply a symmetry-respecting polynomial basis to a
nonlinear energy model. They do not assign a fixed exchange, DMI, anisotropy,
or other spin-Hamiltonian coefficient to any individual descriptor channel.

Symmetries
==========

All Spin3 descriptor coordinates are even under global time reversal,

.. math::

   \boldsymbol{s}_i\longrightarrow-\boldsymbol{s}_i.

With ``spin_soc 0``, the spatial vectors and spin vectors may be rotated
independently and the descriptor is invariant under both rotations. With
``spin_soc 1``, the extra channels are invariant under a joint orthogonal
transformation of the lattice vectors and axial spin vectors,

.. math::

   \boldsymbol{e}_{ij}\longrightarrow R\boldsymbol{e}_{ij},
   \qquad
   \boldsymbol{s}_i\longrightarrow\det(R)R\boldsymbol{s}_i,
   \qquad R\in O(3).

The :ref:`spin_dof_type <kw_spin_dof_type>` mask selects center atoms that
receive public magnetic forces. The
:ref:`spin_env_type <kw_spin_env_type>` mask selects neighbor types that enter
the moment sums. An environment-only spin can therefore affect the energy and
other spins while its public magnetic force is zero. This mask belongs to the
potential response and is not an integration constraint; a TSPIN integrator
still propagates that spin coordinate.

.. _nep_spin_dimensions:

Model dimensions
================

Let :math:`P=C(C+1)/2`, :math:`I_X` be 1 when condition :math:`X` is true and
0 otherwise, and define

.. math::

   a_C=1+I_{C\geq2},\qquad b_C=1+2I_{C\geq2}.

For :math:`L=l_\mathrm{max}` and :math:`S=\texttt{spin\_soc}`, the order-1
Spin3 dimension is

.. math::

   D_1=1+2C+2C I_S I_{L\geq2}.

The order-2 increment is

.. math::

   \Delta D_2 &= 2C
   +C I_{L\geq1}\left(3I_S+1-I_S\right)
   +C I_{L\geq2}+C+2P\\
   &+a_C C I_S I_{L\geq1}
   +a_C C I_S I_{L\geq2}
   +P I_{L\geq1}+P I_{L\geq2},

and the order-3 increment is

.. math::

   \Delta D_3 &= C
   +a_C C I_S I_{L\geq1}\\
   &+b_C C I_S I_{L\geq2}
   +C I_S I_{L\geq1} I_{C\geq3}.

Therefore :math:`D_2=D_1+\Delta D_2` and
:math:`D_3=D_2+\Delta D_3`. For the common :math:`L=2`, ``spin_soc 1`` shape,
O2-C2 has 43 magnetic coordinates and O3-C2 has 55. The implementation limits
the magnetic part to 96 coordinates and requires the combined structural and
magnetic dimension to satisfy :math:`D_\mathrm{str}+D_\mathrm{spin}\leq199`.
With the ordinary structural defaults, :math:`D_\mathrm{str}=42`; the default
O3-C2-L2-SOC1 shape therefore gives :math:`42+55=97` total coordinates.

The trainable magnetic descriptor parameters comprise
:math:`9C N_\mathrm{typ}^2` radial coefficients and, for Spin3,
:math:`4C^2` radial-leg mixing coefficients. These are optimized together
with the neural-network parameters by SNES.

Structural coordinates retain the ordinary inverse-range scaler. Magnetic
coordinates use
:math:`1/\max(\operatorname{RMS}(q),\operatorname{range}(q),1)`, without
centering. Their gain is therefore never larger than one, including for a
constant or initially unresolved magnetic channel.
For fresh training, :ref:`initial_para <kw_initial_para>` scales the radial
coefficients of the fixed reference model used to compute these scalers.
Its default of 1 retains the reference unchanged. Prediction, restart, and
explicit scaler import reuse the stored normalization instead.

Energy derivatives
==================

The atomic force and the unmasked magnetic derivative are defined by

.. math::

   \boldsymbol{F}_k=-\frac{\partial U}{\partial\boldsymbol{r}_k},
   \qquad
   \widetilde{\boldsymbol{M}}_k
   =-\frac{\partial U}{\partial\boldsymbol{s}_k}.

For an atom type enabled by :ref:`spin_dof_type <kw_spin_dof_type>`, the
public magnetic force is
:math:`\boldsymbol{M}_k=\widetilde{\boldsymbol{M}}_k`. For an inactive type it
is explicitly set to zero, even when that atom is retained as an
environment-only spin input. This does not remove the atom from the spin
integrator or its thermostat degrees of freedom.

Writing :math:`q^i_\nu` for either a structural or magnetic descriptor
component, the chain rule gives

.. math::

   \boldsymbol{F}_k
   =-\sum_{i,\nu}
     \frac{\partial U_i}{\partial q^i_\nu}
     \frac{\partial q^i_\nu}{\partial\boldsymbol{r}_k}
     -\frac{\partial U_\mathrm{ZBL}}{\partial\boldsymbol{r}_k},

.. math::

   \widetilde{\boldsymbol{M}}_k
   =-\sum_{i,\nu}
     \frac{\partial U_i}{\partial q^i_\nu}
     \frac{\partial q^i_\nu}{\partial\boldsymbol{s}_k}.

The implementation first evaluates
:math:`\partial U_i/\partial q^i_\nu` through the neural network and then
applies the analytical vector--Jacobian product of every Spin3 contraction,
moment reduction, and radial basis function. The same position derivative
produces the full per-atom virial. No finite-difference force path is used.

The spin torque used for training diagnostics is

.. math::

   \boldsymbol{\tau}_i
   =\boldsymbol{s}_i\times\boldsymbol{M}_i.

ZBL contributes to energy, atomic force, and virial but not to magnetic force
or torque because it is spin independent.

Pressure-controlled spin dynamics
=================================

``npt_tspin`` combines TSPIN with the same Nosé--Hoover-chain and
Parrinello--Rahman/MTTK cell dynamics used by :attr:`npt_mttk`. Its extended
energy has the schematic form

.. math::

   \mathcal{H}_\mathrm{ext}
   = K_R + K_S + U(\boldsymbol R,\boldsymbol S)
   + P_\mathrm{ext}V
   + \mathcal{H}_{\mathrm{NH},R}
   + \mathcal{H}_{\mathrm{NH},S}
   + \mathcal{H}_{\mathrm{NH},h},

where :math:`K_R` is lattice kinetic energy,
:math:`K_S=\sum_i|\boldsymbol\pi_i^S|^2/(2\mu_i)` is spin-coordinate kinetic
energy, and :math:`h` denotes the cell degrees of freedom. The spin equations
are

.. math::

   \dot{\boldsymbol s}_i
   = \frac{\boldsymbol\pi_i^S}{\mu_i},
   \qquad
   \dot{\boldsymbol\pi}_i^S
   = \boldsymbol M_i-\xi_S\boldsymbol\pi_i^S.

Cartesian spins are internal coordinates: changing :math:`h` remaps atomic
positions, but neither rescales nor rotates :math:`\boldsymbol s_i`. Therefore
the barostat pressure contains lattice kinetic stress and the full potential
virial,

.. math::

   P_{\alpha\beta}
   = \frac{1}{V}\left[
       \sum_i m_i v_{i\alpha}v_{i\beta}
       + W_{\alpha\beta}(\boldsymbol R,\boldsymbol S)
     \right],

but no :math:`K_S` term. Spin still affects the cell because the analytical
virial :math:`W` differentiates the complete Spin-potential energy with
respect to cell deformation. Separate Nosé--Hoover chains control lattice and
spin kinetic energies at the same target-temperature ramp; the pressure chain
controls the cell momenta.

The complete user-facing syntax, initialization rules, restart behavior, and
output example are documented under :ref:`npt_tspin <kw_npt_tspin>`. This
implementation does not define ``npt_ber_tspin`` or ``npt_scr_tspin``; the
only pressure-controlled TSPIN path is the MTTK composition described above.

Spin loss terms
===============

Spin training augments the ordinary :ref:`NEP loss <nep_loss_function>` with
magnetic-force and torque RMSE terms,

.. math::

   L_\mathrm{spin}
   = \lambda_\mathrm{m} L_\mathrm{m}
   + \lambda_\mathrm{tau} L_\mathrm{tau}
   + \lambda_\mathrm{resp} L_\mathrm{resp},

where :math:`N_\mathrm{m}` counts labeled active spins and
:math:`N_\mathrm{tau}` counts only their nonzero spins,

.. math::

   L_\mathrm{m}
   &=\left[
     \frac{1}{3N_\mathrm{m}}
     \sum_i |\boldsymbol{M}^{\mathrm{NEP}}_i-
                  \boldsymbol{M}^{\mathrm{tar}}_i|^2
     \right]^{1/2},\\
   L_\mathrm{tau}
   &=\left[
     \frac{1}{2N_\mathrm{tau}}
     \sum_i |\boldsymbol{s}_i\times\boldsymbol{M}^{\mathrm{NEP}}_i-
                  \boldsymbol{s}_i\times\boldsymbol{M}^{\mathrm{tar}}_i|^2
     \right]^{1/2}.

The magnetic-force loss always includes all three Cartesian components and
includes zero spins. The torque loss excludes zero spins. To supervise only
spin rotation, set ``lambda_m 0`` and a positive ``lambda_tau``. For nonzero
spins, the squared torque error equals the squared transverse magnetic-force
error multiplied by :math:`|\boldsymbol{s}_i|^2`.

For a grouped rotation path with coordinate :math:`\theta`, the response
generator evaluated for frame :math:`n` is

.. math::

   G_n
   =\sum_i \boldsymbol{M}_{ni}\cdot
      \frac{\partial\boldsymbol{s}_{ni}}{\partial\theta}
   =-\frac{\partial U_n}{\partial\theta}.

The grouped response loss first centers :math:`G_n` within each group. Each
group contributes equally to the centered-curve Huber loss, independent of its
number of scan points. Each group's uncentered target-response RMS, floored
at :math:`10^{-6}`, normalizes its shape term and a second Huber term comparing
group means; the mean term is multiplied by 0.25. ``response_coordinate``
defines the path derivative but does not add a separate reliability weight.
This GPUMD policy differs from the older shared-scale reduction.

The group coordinate is explicit because Cartesian spin snapshots alone do
not determine their path ordering or parameterization. The trainer sorts each
group and derives
:math:`\partial\boldsymbol{s}/\partial\theta` from the final constrained-DFT
spins with second-order, three-point nonuniform weights applied to sphere-log
angular displacements and separate magnitude differences. This preserves
longitudinal moment relaxation without injecting the radial chord error of
Cartesian interpolation into constant-length rotations. Stencils must resolve
angular separations below pi. Atom types excluded by ``spin_dof_type`` are
masked from the derived tangent. See :ref:`lambda_spin_response
<kw_lambda_spin_response>` for zero-moment and finite-stencil limitations.

Spin3 uses the shared SNES lifecycle: generation zero prepares training scalers
and uses regularization-only fitness. Physical losses are evaluated from the
next generation. All magnetic descriptor orders participate throughout training.

Model formats and input
=======================

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Header in ``nep.txt``
     - Training mode
     - Description
   * - ``nep`` / ``model nep_spin``
     - ``spin_version 1``
     - Current block format with ``spin_version 1``; optional :ref:`zbl <kw_zbl>` field.

See :ref:`spin_version <kw_spin_version>` and the neighboring ``spin_*`` keyword
pages for descriptor configuration. The complete extended-XYZ data contract,
including magnetic-force labels and grouped response metadata, is documented
under :ref:`train_test_xyz`.
Runtime ``model.xyz`` files require one ``spin:R:3`` vector per atom.
TSPIN dynamics is documented in :ref:`kw_ensemble_tspin`.
Fixed-magnitude stochastic dynamics is documented in :ref:`kw_ensemble_sib`.
