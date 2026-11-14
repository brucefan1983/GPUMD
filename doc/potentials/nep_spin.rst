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
with ``spin_version 1`` and named parameter and scaler blocks. It uses the
tensor descriptor described below. An optional ``zbl`` field adds a spin-independent
universal ZBL term. Flexible ZBL (``zbl 0 0``) is not supported.
``spin_version`` identifies the descriptor and parameter-layout
contract, independently of the file syntax version and ``spin_order``.

Every :ref:`spin_cutoff <kw_spin_cutoff>` value must be positive and no larger
than the ordinary NEP radial cutoff. This constraint applies to both training
input and model files; different spin cutoffs per atom type are supported
within this bound.

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
magnetic coordinates. Therefore, ``spin_order`` describes the maximum degree of
primitive spin polynomials supplied to the network; it is not the body
order of the final energy.

.. _nep_spin_contractions:

Spin descriptors
================

Let :math:`C=\texttt{spin_compress}`, :math:`L=\texttt{spin_l_max}` and
:math:`S=\texttt{spin_order}`. Real, component-normalized solid harmonics
form the spin primitives

.. math::

   \phi_{dl}(\mathbf{s}) = |\mathbf{s}|^{d-l}
      \mathcal{Y}_l(\mathbf{s}),
   \qquad 0\leq d\leq S,\quad l=d\bmod2,d\bmod2+2,\ldots,\min(d,L).

In SOC mode (``spin_soc 1``), the angular NEP radial coefficients are reused at the spin cutoff, then
linearly mixed into :math:`C` channels. For each allowed geometry rank,
spin primitive and coupled output rank, an independent learned channel
weight multiplies the real Clebsch--Gordan coupling. Degree-zero primitives
use one site; positive-degree primitives use both center and neighbor spins.
Edges are summed into center densities grouped by output rank, spatial
parity and time-reversal parity. Learned center-only spin primitives are
added to the matching sectors, including for isolated atoms.

The response coordinates are the local positive-degree scalar spin
primitives, the even scalar tensor sum itself, and each sector's squared norm
divided by :math:`2l+1`, separately for each channel. These coordinates are
concatenated with the unchanged structural NEP coordinates. This retains
simultaneous rotation, inversion and time-reversal invariance. Cartesian
force, magnetic force and all nine virial components are analytical
chain-rule derivatives of this energy.

Parameter and checkpoint layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The shared block-file syntax remains ``format_version 1``. The magnetic
layout is determined by ``spin_soc`` and the descriptor settings. Parameters are stored
as per-type ANN parameters, shared ANN bias, structural radial coefficients,
structural angular coefficients, then these channel-last arrays:

* angular-radial mixing, shape ``(n_max_angular+1, C)``;
* coupling-path weights, shape ``(number_of_paths, C)``;
* center-primitive mixing, shape ``(number_of_positive_degree_primitives, C)``.

The sizes of these arrays follow from the descriptor settings. Primitive order is
ascending degree then rank. In SOC mode, path order is geometry rank, primitive, site
(center before neighbor), then output rank. Sectors are sorted by
``(rank, spatial parity, time parity)``, with -1 before +1. Scalers follow
in the ``@ q_scaler`` block.

Current limits are :math:`1\leq C\leq9`, :math:`0\leq L\leq4`,
:math:`1\leq S\leq4`, at most 96 response and 199 total coordinates.
The training optimizer is SNES.

Non-SOC descriptors
^^^^^^^^^^^^^^^^^^^

Set ``spin_soc 0``. The radial mix, variable-length
spin primitives, and energy-center mask semantics are shared with the SOC
mode. Instead of coupling spatial rank g and spin rank j into a joint rank,
keep the product components (m,a) separate::

  T[c,g,j,m,a] = sum_edges sum_degrees sum_sites
                  radial[c](r) * weight[c,g,j,degree,site]
                  * Y[g,m](r_hat) * Phi[degree,j,a](spin_site)
                + delta[g,0] * sum_degrees center[c,degree]
                  * Phi[degree,j,a](spin_center)

The scalar responses are the full squared Frobenius norms divided by
``(2*g+1)*(2*j+1)``. Also retain the direct ``g=j=0`` tensor sum and local
even-degree scalar spin primitives. Degrees of the same spin rank mix;
center and neighbor spins can interfere within a product sector, preserving
exchange-sensitive responses. Rotating only one spin generally changes the
energy, while rotating every spin by the same orthogonal transformation does
not. No normalization by spin length is required at zero spin.

Product sectors are ordered by ascending ``(g,j)``. Paths are ordered by
``g, j, primitive (degree then rank), site (center then neighbor)``; within
a product sector the spatial component m precedes spin component a. Center
coefficients keep the positive-degree primitive ordering of the SOC mode,
but only enter spatial-rank-zero sectors. This ordering is part of the
``spin_soc 0`` checkpoint contract.

.. _nep_spin_dimensions:

For maximum angular rank L, order S and C channels, the non-SOC dimension is
``floor(S/2) + C * ((L+1)*(min(L,S)+1) + 1)``. For C=3, L=S=2 this is 31,
with 81 tensor components per channel. The existing limits of 96 response
and 199 total coordinates still apply. The largest supported product basis
has 625 components per channel; large layouts retain the generic CUDA
fallback. SOC at L=S=2 has ``1+11*C`` spin coordinates.

.. _nep_spin_scaling:

Descriptor scaling
------------------

Structural coordinates retain the ordinary inverse-range scaler. Magnetic
coordinates use
:math:`1/\max(\operatorname{RMS}(q),\operatorname{range}(q),1)`, without
centering. Their gain is therefore never larger than one, including for a
constant or initially unresolved magnetic channel.

Fresh training computes scaling from ordinary training structures using a
fixed, non-degenerate reference parameter vector, independent of the SNES
search mean and RNG. Nonconstant spin projection matrices keep descriptor
channels distinct. Response-only frames do not enter this calculation.
Prediction, restart, and explicit scaler import reuse the stored normalization
instead.

.. _nep_spin_restart:

Restart behavior
----------------

Spin restart restores the scaler from the matching ``nep.txt``. See
:ref:`import_q_scaler <kw_import_q_scaler>` for automatic restoration and
fresh scaler-only import. The two files must have compatible architecture and
element order, and the scaler must be finite and positive.

``nep.restart`` stores the search distribution and ``energy_reference`` in the
shared restart format, while ``nep.txt`` stores a selected candidate model.
The search mean and that candidate need not predict the same values. Restart
does not rescale parameters or substitute the candidate for the search mean.

A restart begins a new requested generation segment and resets RNG
initialization. The restart file does not store global generation progress;
it is not an exact replay of an uninterrupted trajectory.

Energy derivatives
==================

The atomic force and the unmasked magnetic derivative are defined by

.. math::

   \boldsymbol{F}_k=-\frac{\partial U}{\partial\boldsymbol{r}_k},
   \qquad
   \widetilde{\boldsymbol{M}}_k
   =-\frac{\partial U}{\partial\boldsymbol{s}_k}.

For an atom type enabled by :ref:`spin_type <kw_spin_type>`, the
public magnetic force is :math:`\boldsymbol{M}_k=\widetilde{\boldsymbol{M}}_k`.
Excluded types have no independent spin coordinate; their public magnetic force
is zero by convention. They still receive atomic forces from the positional
derivative of the full energy, including magnetic geometric environments.

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
applies the analytical vector--Jacobian product of every NEP-spin contraction,
moment reduction, and radial basis function. The same position derivative
produces the full per-atom virial. No finite-difference force path is used.

The spin torque used for training diagnostics is

.. math::

   \boldsymbol{\tau}_i
   =\boldsymbol{s}_i\times\boldsymbol{M}_i.

ZBL contributes to energy, atomic force, and virial but not to magnetic force
or torque because it is spin independent.

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
:math:`N_\mathrm{tau}` counts only those with
:math:`|\boldsymbol{s}_i|^2 > 10^{-20}\,\mu_\mathrm{B}^2`,

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
includes zero spins. Both the torque-error sum and its count exclude spins at
or below the squared-magnitude threshold, evaluated in double precision.
To supervise only spin rotation, set ``lambda_m 0`` and a positive ``lambda_tau``.
For nonzero spins, the squared torque error equals the squared transverse magnetic-force
error multiplied by :math:`|\boldsymbol{s}_i|^2`.

.. _nep_spin_response:

Grouped magnetic response
-------------------------

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

The group coordinate is explicit because Cartesian spin snapshots alone do
not determine their path ordering or parameterization. The trainer sorts each
group and derives
:math:`\partial\boldsymbol{s}/\partial\theta` from the final constrained-DFT
spins with second-order, three-point nonuniform weights applied to sphere-log
angular displacements and separate magnitude differences. This preserves
longitudinal moment relaxation without injecting the radial chord error of
Cartesian interpolation into constant-length rotations. Stencils must resolve
angular separations below pi. Atom types excluded by ``spin_type`` are
masked from the derived tangent. At zero moment, the tangent uses Cartesian
differences. Constant-rate great-circle rotations are differentiated exactly
apart from floating-point rounding; other paths retain the finite-stencil
approximation. Antipodal samples require intermediate frames. Input labels
and group metadata are specified in :ref:`response_xyz`; the loss weight is
set by :ref:`lambda_spin_response <kw_lambda_spin_response>`.

NEP-spin uses the shared SNES lifecycle: generation zero prepares training scalers
and uses regularization-only fitness. Physical losses are evaluated from the
next generation. All magnetic descriptor orders participate throughout training.

Both native and runtime-compiled NEP-spin training evaluate ANN dot products,
activations, energy sums, and descriptor-gradient accumulation in double
precision. This limits cancellation errors in small magnetic energy differences
while keeping the energy and its derivatives on the same arithmetic path.
Model parameters and ANN descriptor gradients are stored in single precision.
The spin neighbor list uses the original double-precision positions and cell.
For non-SOC models with spin angular rank four and spin order four, the spin
descriptor and force kernels share double-precision contracted radial
coefficients.

Model formats and input
=======================

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Header in ``nep.txt``
     - Training mode
     - Description
   * - ``nep`` / ``model nep_spin``
     - ``model nep_spin``
     - Current block format with ``spin_version 1``; optional :ref:`zbl <kw_zbl>` field.

See :ref:`spin_version <kw_spin_version>` and the neighboring ``spin_*`` keyword
pages for descriptor configuration. See :ref:`train_test_xyz` for ordinary
training labels and :ref:`response_xyz` for grouped-response input.

.. _nep_spin_runtime:

Fixed-spin molecular dynamics
-----------------------------

Load the exported model with ``potential nep.txt`` and provide Cartesian
magnetic moments through ``spin:R:3`` in :ref:`model.xyz <model_xyz>`.
The ``nve`` and ``nvt_nhc`` ensembles evolve positions and velocities while
keeping the magnetic moments fixed. Only one potential is allowed, and all
three cell directions must be periodic.

MD stores and integrates Cartesian positions in double precision. Most NEP-spin
layouts convert relative displacements to single precision for descriptor and
force evaluation. The non-SOC rank-four, order-four spin kernels evaluate
these displacements in double precision. Use ``dump_xyz ... precision double``
to retain coordinate precision in trajectory output.

Atoms whose elements are absent from ``spin_type`` have zero effective spin.
Their input spin values are ignored; they still contribute geometric
environments. Their magnetic forces remain zero, including in an applied
external field. The reader does not normalize the active spins.

For example::

   potential nep.txt
   velocity 300
   ensemble nve
   time_step 1
   dump_xyz 100 trajectory.xyz spin mforce
   dump_restart 1000
   run 1000

Use ``dump_xyz`` with ``spin`` and ``mforce`` to write the moments and magnetic
forces. ``dump_restart`` preserves the fixed spins in ``restart.xyz``;
copy this file to ``model.xyz`` and use the same potential to resume.
``replicate`` copies the spin vectors with their atoms.

The :ref:`add_mfield keyword <kw_add_mfield>` applies a magnetic induction in
tesla to a group or fractional-coordinate region. Magnetic force is
:math:`-\partial E/\partial \mathbf{s}` in eV/Bohr magneton; it is evaluated
although spins are fixed. For spin dynamics, see :ref:`SIB <kw_ensemble_sib>`,
:ref:`TSPIN <kw_ensemble_tspin>`, and :ref:`GLSD <kw_ensemble_glsd>`.

Multi-GPU execution uses every visible GPU in a single process. Use
``potential nep.txt x`` (or ``y``/``z``) to select the spatial partition
direction. Small cells may require replicated geometry and offer little
speedup. GPU 0 retains the global arrays.
