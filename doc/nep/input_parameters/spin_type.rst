.. _kw_spin_type:
.. index::
   single: spin_type (keyword in nep.in)

:attr:`spin_type`
======================

Select the atom types that have independent spin degrees of freedom::

  spin_type Fe

At least one type must be listed. Names must be unique and must appear in
:ref:`type <kw_type>`. If omitted, all atom types are selected. The effective
selection is saved in the model and restored for prediction and restart;
an explicit conflicting selection is rejected.

Selected types are magnetic descriptor centers and contribute to magnetic-force,
parallel/perpendicular and torque supervision when labels are present. All atom
types contribute structural and magnetic geometric environments within the
corresponding cutoffs. Only selected types contribute neighbor spin moments.

For atoms outside this selection, the reader sets the effective spin to zero
and ignores magnetic-force labels. Original data files need no modification;
energy and atomic-force labels are preserved. Their public magnetic force is
zero and they are excluded from magnetic loss counts. Their positions and
atomic forces remain active, including forces due to magnetic energy.

The grouped-response path derivative uses the selected DFT spin coordinates.
With :ref:`lambda_spin_response <kw_lambda_spin_response>`, derived tangents
are zero on excluded types.
