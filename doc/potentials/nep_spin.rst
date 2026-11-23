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
       + E^{\mathrm{wall}}_{Z_i}(|\boldsymbol{s}_i|)
     \right] + U_\mathrm{ZBL},

where :math:`\boldsymbol{r}_i` is the position, :math:`\boldsymbol{s}_i` is a
Cartesian magnetic-moment vector in Bohr magnetons (muB), :math:`Z_i` denotes the atom type, and
:math:`E^0_{Z_i}` is a fitted per-type energy baseline. The ZBL term is present
when ZBL is enabled and contains no spin dependence. The reference energies
use the shared ``energy_reference`` convention. The optional
:ref:`spin_wall <kw_spin_wall>` term :math:`E^{\mathrm{wall}}_{Z_i}` is a
fixed analytic function of the moment magnitude. Its negative spin gradient
contributes a longitudinal magnetic force. Training, prediction, and MD all
include these contributions. They vanish for disabled walls and elements
without active spin degrees of freedom. The wall adds no descriptor
components or trainable parameters.

This architecture allows the nonlinear network to combine structural and
magnetic coordinates. Therefore, ``spin_order`` describes the maximum polynomial degree of
primitive spin inputs supplied to the network; it is not the body
order of the final energy.

.. _nep_spin_contractions:

Magnetic descriptor
===================

The descriptor combines the central spin, neighboring spins, and local atomic geometry within :ref:`spin_cutoff <kw_spin_cutoff>`.
It forms scalar invariants from local spin and geometric densities. The main choices are:

* :ref:`spin_compress <kw_spin_compress>` sets the number of learned radial channels.
* :ref:`spin_order <kw_spin_order>` sets the maximum polynomial degree of the spin inputs, including changes in moment magnitude.
* :ref:`spin_l_max <kw_spin_l_max>` bounds the spatial and spin tensor ranks and, in SOC mode, the coupled rank.
* :ref:`spin_soc <kw_spin_soc>` controls whether spin orientation couples to the local lattice orientation.
* :ref:`spin_type <kw_spin_type>` selects the elements with active spin degrees of freedom.

The spin descriptor reuses the structural angular radial coefficients with its own magnetic cutoff.
The second :ref:`basis_size <kw_basis_size>` value also sets its radial basis; ``basis_size 8 8`` gives nine basis functions.
The default spin settings are ``spin_compress 3``, ``spin_order 2``, ``spin_l_max 2``, and ``spin_soc 1``.
Spin vectors are not normalized, and zero moments are supported.

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

All atom types are energy centers. The :ref:`spin_type <kw_spin_type>`
selection defines the independent spin coordinates. All neighbor types enter geometric
moment sums within the spin cutoff. The reader sets excluded types' effective
spins to zero and ignores their magnetic-force labels, so their contributions
to moments containing neighbor spins vanish while geometric contributions remain.
Original energy and atomic-force labels are retained; input files are unchanged.

.. _nep_spin_dimensions:

Model dimensions
================

The magnetic dimension depends on :ref:`spin_compress <kw_spin_compress>`,
:ref:`spin_order <kw_spin_order>`, :ref:`spin_l_max <kw_spin_l_max>`, and :ref:`spin_soc <kw_spin_soc>`.
The default ``spin_compress 3``, ``spin_l_max 2``, ``spin_order 2`` shape has
34 magnetic coordinates with SOC and 31 without SOC.

All combinations of ``spin_compress 1--9``, ``spin_l_max 0--4``, ``spin_order 1--4``, and ``spin_soc 0/1`` are supported.
The limits are 236 magnetic and 339 combined structural and magnetic coordinates; the structural angular limit remains 90.

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

.. _spin_full_virial:

Full virial supervision
=======================

NEP-Spin takes the full cell tensor from the nine-component :ref:`virial field <train_test_xyz>`.
The deformation gradient :math:`F` acts on the cell and atomic positions while the Cartesian spins are held fixed.
The virial is :math:`W_{ij}=-\left.\partial U/\partial F_{ji}\right|_{F=I}`, evaluated at the undeformed configuration.
The full tensor includes the structural, spin and ZBL derivatives. Only the spin term can contribute to the antisymmetric part of the cell virial.
Rotational invariance of the joint spin and lattice rotation gives

.. math::

   (W_{yz}-W_{zy},\, W_{zx}-W_{xz},\, W_{xy}-W_{yx})
   = -\sum_i \boldsymbol{s}_i\times\boldsymbol{M}_i.

The :ref:`lambda_v <kw_lambda_v>` objective uses all nine Cartesian residuals of the per-atom virial for every labeled frame, including symmetric and zero tensors.
The RMSE divides the sum of squared residuals by nine times the number of frames with a ``virial`` label; unlabeled frames do not contribute.
For the training loss, the structure ``weight`` multiplies every residual and :ref:`lambda_shear <kw_lambda_shear>` also multiplies each off-diagonal residual before squaring.

Spin loss terms
===============

Spin training augments the ordinary :ref:`NEP loss <nep_loss_function>` with
magnetic-force, torque, and grouped-response terms.
With the default Cartesian magnetic-force objective controlled by :ref:`lambda_m <kw_lambda_m>`,

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

This Cartesian magnetic-force RMSE includes all three components and zero spins.
Both the torque-error sum and its count exclude spins at
or below the squared-magnitude threshold.
To supervise only spin rotation, set ``lambda_m 0`` and a positive ``lambda_tau``.
For nonzero spins, the squared torque error equals the squared transverse magnetic-force
error multiplied by :math:`|\boldsymbol{s}_i|^2`.

To weight the parallel and perpendicular magnetic-force components separately, replace ``lambda_m``
with both :ref:`lambda_m_parallel and lambda_m_perp <kw_lambda_m_parallel>`.
Either component can be disabled by setting its coefficient to zero.

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
Parameters from the previous Spin3 descriptor layout are incompatible with this model.
See :ref:`spin_version <kw_spin_version>` for the training settings and compatibility requirements.
The parameter and scaler layouts are specified in
`the NEP model format <https://github.com/brucefan1983/GPUMD/blob/master/potentials/nep/nep_model_format.md>`_.

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
