.. _response_xyz:

``response.xyz``
================

Place this file beside ``nep.in`` when enabling
:ref:`lambda_spin_response <kw_lambda_spin_response>`. It contains complete
rotation paths for the magnetic-response objective, independently of the
ordinary training minibatches. With zero response weight the file is ignored.

Frame format
------------

Use extended XYZ frames with a cell (``Lattice``), ``species:S:1``, ``pos:R:3``,
``energy``, ordinary ``force:R:3``, final constrained-DFT ``spin:R:3`` and
``mforce:R:3``. All these labels are required; only virial is optional.
Missing energy or force is an input error, not a zero-valued label. Use the
converged spins, not the initially requested constraint vectors. Attribute
aliases, units, and :ref:`spin_type <kw_spin_type>` masking follow
:ref:`train_test_xyz`.

Each frame also needs line-2 metadata::

   response_probe=rotation response_group=scan-a response_coordinate=0.0

``response_group`` identifies one physical rotation path.
``response_coordinate`` orders its frames and parameterizes the path; for a
rotation angle, use radians. Coordinates need only be comparable within a
group. Each group must have at least three finite, distinct coordinates and
fixed atom order, species, positions, and cell. The trainer sorts each group
by coordinate and derives tangents internally from the final spins; do not
supply ``spin_tangent``.

Resolve angular separations within each three-point stencil below pi;
antipodal samples require intermediate frames. See
:ref:`nep_spin_response` for the tangent definition and finite-stencil limits.
Ordinary spin training does not require response metadata. Non-``rotation``
probe metadata in ordinary train/test data is accepted for analysis;
``response.xyz`` requires rotation groups.

Data separation
---------------

Response-only frames never enter ordinary energy, force, virial,
magnetic-force or torque losses, baseline fitting, or descriptor scaling.
Energy and ordinary force labels are required for complete frames but do not
enter the response loss. A frame contributes to ordinary losses only if it
is explicitly included in ``train.xyz``. Group weighting and the response
objective are described under :ref:`lambda_spin_response <kw_lambda_spin_response>`.

Keep response and validation data disjoint. With response loss enabled,
shared response group names between train/test or response/test are rejected.
An identical stored cell, species/order, positions and spin in response/test
is also rejected, even if the group name is removed or changed. This exact
frame check does not identify atom permutations or periodic-image equivalents;
keep complete physical scans in one split when preparing the data.
