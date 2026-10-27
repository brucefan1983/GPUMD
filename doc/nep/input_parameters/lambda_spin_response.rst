.. _kw_lambda_spin_response:
.. index::
   single: lambda_spin_response (keyword in nep.in)

:attr:`lambda_spin_response`
=============================

This keyword sets the weight of grouped magnetic-response loss::

  lambda_spin_response <weight>

The weight must be finite and non-negative and defaults to 0. A positive value
requires ``spin_version 1`` and :ref:`response.xyz <response_xyz>` in the working
directory. A missing file, an empty file, or an incomplete rotation group is
an input error. With weight 0, the file is not opened.

Ordinary ``train.xyz`` data still use the ``batch`` setting. All complete
response groups are evaluated separately for every candidate in every SNES
generation after initialization (generation 0). As in ordinary NEP, generation 0
prepares training state without physical candidate fitness. Response groups are never split by ordinary minibatches. There
is no ``response_file`` keyword or additional response batch setting.

Response frames contribute only this loss. They do not contribute ordinary
energy, force, virial, magnetic-force or torque losses, the fitted energy
baseline, or descriptor scaling. They reuse the baseline and scaler from
``train.xyz``. GPUMD's existing test reporting and model-selection behavior
are unchanged; no separate response validation loss is added.

For a response frame :math:`a`, the derived tangent
:math:`\boldsymbol{t}_{ai}` defines a scalar generalized response

.. math::

   R_a=\sum_i \boldsymbol{M}_{ai}\cdot\boldsymbol{t}_{ai}.

The loss compares predicted and target :math:`R_a` values within each named
group. It robustly constrains both the centered response shape along the path
and, with a smaller weight, the group mean. This is useful when relative
magnetic response along a controlled rotation path matters in addition to
per-component magnetic-force RMSE.

Every ``response.xyz`` frame must provide ``energy``, ``force:R:3``, the
converged DFT ``spin:R:3`` and ``mforce:R:3`` (virial is optional), together
with::

  response_probe=rotation response_group=<name> response_coordinate=<value>

Each group needs at least three distinct ``response_coordinate`` values, and
the trainer sorts each complete group by this coordinate and derives
``d spin / d response_coordinate`` using second-order, three-point nonuniform
weights in local spin coordinates. Angular displacements use the sphere
logarithm at the current spin; magnitude differences are differentiated
separately. This avoids a spurious radial tangent for a fixed-length rotation.
The coordinate does not have to be globally comparable between groups.

For a rotation angle :math:`\theta`, ``response_coordinate`` should be
:math:`\theta` in radians. ``spin`` must contain the moments from the final
constrained-DFT state, not the initially requested constraint vectors. The
derived path tangent retains both transverse and longitudinal relaxation.
Atoms excluded by :ref:`spin_type <kw_spin_type>` are masked from the
derived response automatically. ``spin_tangent:R:3`` is not accepted as an
input label.

The rotation path must be resolved: angular separations within a stencil must
be less than pi. Antipodal samples are ambiguous and require intermediate
frames. At zero moment the tangent uses Cartesian differences. Constant-rate
great-circle rotations are differentiated exactly apart from floating-point
rounding; other paths retain the finite-stencil approximation.

With this tangent convention, :math:`R_a\approx-\partial U_a/\partial\theta`.
Each group contributes equally and uses its own uncentered target-response
RMS, with a floor of :math:`10^{-6}`, to normalize its centered-shape and
group-mean Huber terms. The latter has weight 0.25. A large exchange response
in one group therefore cannot suppress a small anisotropy response in another.
Coordinates do not create an additional reliability weight.
