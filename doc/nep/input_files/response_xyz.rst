.. _response_xyz:

``response.xyz``
================

Place this file beside ``nep.in`` when enabling
:ref:`lambda_spin_response <kw_lambda_spin_response>`. It contains complete
rotation paths for the magnetic-response objective, independently of the
ordinary training minibatches. With zero response weight the file is ignored.

Use extended XYZ frames with a cell (``Lattice``), ``species:S:1``, ``pos:R:3``,
``energy``, ordinary ``force:R:3``, final constrained-DFT ``spin:R:3`` and
``mforce:R:3``. All these labels are required, as in complete magnetic
training data; only virial is optional. Missing energy or force is an input
error, not a zero-valued label. Each frame also needs line-2 metadata::

   response_probe=rotation response_group=scan-a response_coordinate=0.0

Each group must have at least three finite, distinct coordinates and fixed
atom order, species, positions, and cell. For a rotation angle, use radians.
Tangents are derived internally from the final spins; do not supply
``spin_tangent``. Spin is measured in Bohr magnetons and magnetic force in
eV per Bohr magneton.

All groups enter each candidate's response loss in every generation with equal
group weight. Response-only frames never enter the ordinary energy, force,
virial, magnetic-force or torque losses. Baseline fitting and descriptor
scaling use only ordinary training data. A frame contributes to ordinary
losses only if explicitly included in ``train.xyz``. Requiring complete
labels does not enable additional response supervision: energy and ordinary
force labels are currently unused by the response loss, and no relative-energy
response loss is implemented.

Keep response and validation data disjoint. With response loss enabled,
shared response group names between train/test or response/test are rejected.
An identical stored cell, species/order, positions and spin in response/test
is also rejected, even if the group name is removed or changed. This exact
frame check does not identify atom permutations or periodic-image equivalents;
keep complete physical scans in one split when preparing the data.
