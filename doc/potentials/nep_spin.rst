.. _nep_spin:
.. index::
   single: Spin potential
   single: NEP_Spin
   single: magnetic force

Spin potential
**************

NEP-spin extends the :ref:`NEP4 formalism <nep_formalism>` to a local
energy surface that depends on both atomic positions and Cartesian spin
vectors. The total energy is the only learned scalar output. Atomic forces,
virials, magnetic forces, and spin torques are derivatives of this same energy
and are not produced by independent response networks.

To train a model, start with the :ref:`spin training example <spin_training_example>` and provide the spin labels described in :ref:`train_test_xyz`.
The sections below explain the energy model, descriptor choices, and magnetic losses.
For MD, see :ref:`NEP-spin runtime use <nep_spin_runtime>`.

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

.. _nep_spin_contractions:

Magnetic descriptor
===================

The descriptor combines the central spin, neighboring spins, and local atomic geometry within :ref:`spin_cutoff <kw_spin_cutoff>`.
It uses learned radial channels and scalar contractions of spin and spatial vectors and tensors.
The main choices are:

* :ref:`spin_compress <kw_spin_compress>` sets the number of shared radial channels.
* :ref:`spin_order <kw_spin_order>` selects nested levels of magnetic correlations: direct one-edge terms, two-leg correlations, and three-leg correlations.
* :ref:`spin_l_max <kw_spin_l_max>` selects the maximum spatial tensor rank.
* :ref:`spin_soc <kw_spin_soc>` controls whether spin orientation couples to the local lattice orientation.
* :ref:`spin_type <kw_spin_type>` selects the elements with active spin degrees of freedom.

The radial basis uses the same cutoff-weighted Chebyshev construction as ordinary NEP, with the magnetic cutoff.
The supported :ref:`spin_basis_size <kw_spin_basis_size>` is ``8 0``, giving nine radial basis functions and no separate angular basis.
The default descriptor uses ``spin_order 3``, ``spin_compress 2``, ``spin_l_max 2 0 0``, and ``spin_soc 1``.

Symmetries
==========

All magnetic descriptor coordinates are even under global time reversal,

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

The :ref:`spin_type <kw_spin_type>` selection defines independent spin
coordinates and magnetic descriptor centers. All neighbor types enter geometric
moment sums within the spin cutoff. The reader sets excluded types' effective
spins to zero and ignores their magnetic-force labels, so their contributions
to moments containing neighbor spins vanish while geometric contributions remain.
Original energy and atomic-force labels are retained; input files are unchanged.

.. _nep_spin_dimensions:

Model dimensions
================

Let :math:`P=C(C+1)/2`, :math:`I_X` be 1 when condition :math:`X` is true and
0 otherwise, and define

.. math::

   a_C=1+I_{C\geq2},\qquad b_C=1+2I_{C\geq2}.

For :math:`L=l_\mathrm{max}` and :math:`S=\texttt{spin\_soc}`, the order-1
magnetic descriptor dimension is

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
:math:`9C N_\mathrm{typ}^2` radial coefficients and
:math:`4C^2` radial-leg mixing coefficients. These are optimized together
with the neural-network parameters by SNES.

.. _nep_spin_scaling:

Descriptor scaling
------------------

Structural coordinates retain the ordinary inverse-range scaler. Magnetic
coordinates use
:math:`1/\max(\operatorname{RMS}(q),\operatorname{range}(q),1)`, without
centering. Their gain is therefore never larger than one, including for a
constant or initially unresolved magnetic channel.

Fresh training determines the normalization from the ordinary training structures; response-only frames are excluded.
Prediction, restart, and explicit scaler import reuse the stored normalization.
See :ref:`import_q_scaler <kw_import_q_scaler>` for the required files.

.. _nep_spin_restart:

Restart behavior
----------------

Keep the matching ``nep.txt`` and ``nep.restart`` together when continuing training.
The model provides descriptor normalization, while the restart file provides the search state and energy reference.
Their architecture and element order must match; see :ref:`import_q_scaler <kw_import_q_scaler>`.
A restart begins a new generation segment and is not an exact replay of uninterrupted training.

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
or below the squared-magnitude threshold.
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

The response coordinate specifies the ordering and parameterization of each spin-rotation path.
Provide sufficiently close samples to resolve angular separations below pi; antipodal samples need intermediate frames.
GPUMD estimates the path derivative from neighboring frames and includes changes in spin magnitude.
At zero moment it uses Cartesian differences. Types excluded by ``spin_type`` do not contribute.
See :ref:`response.xyz <response_xyz>` for data preparation and :ref:`lambda_spin_response <kw_lambda_spin_response>` for the loss weight.

Model formats and input
=======================

Select ``model nep_spin`` in ``nep.in``. The model requires NEP4 and one hidden layer.
The exported ``nep.txt`` uses ``format_version 1``, ``model nep_spin``, and ``spin_version 1``.
Universal :ref:`ZBL <kw_zbl>` is optional; flexible ZBL is not supported.
See :ref:`spin_version <kw_spin_version>` for the training settings and compatibility requirements.

.. _nep_spin_runtime:

Molecular dynamics
------------------

Load the exported model with ``potential nep.txt`` and provide Cartesian
magnetic moments through ``spin:R:3`` in :ref:`model.xyz <model_xyz>`.
The ``nve`` and ``nvt_nhc`` ensembles evolve positions and velocities while
keeping the magnetic moments fixed. Only one potential is allowed, and all
three cell directions must be periodic.

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
direction. Small systems may gain little speed from multiple GPUs.
