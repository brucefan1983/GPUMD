.. _kw_spin_dof_type:
.. index::
   single: spin_dof_type (keyword in nep.in)

:attr:`spin_dof_type`
======================

This keyword selects atom types for magnetic-force output and magnetic supervision::

  spin_dof_type <type_1> [<type_2> ...]

At least one type must be listed. Names must be unique and must appear in
:ref:`type <kw_type>`. If the keyword is absent, all atom types are selected.

Selected types receive learned magnetic-force output and contribute to Cartesian
magnetic-force and torque RMSEs, including the optional parallel/perpendicular
split, when the corresponding labels are present. Excluded types receive zero
public magnetic force and do not enter these RMSE counts. Their current spins
can nevertheless contribute to a magnetic environment when explicitly included
by :ref:`spin_env_type <kw_spin_env_type>`.

This keyword does not freeze spins during MD or remove them from a thermostat.
Excluded spins can still move because of spin velocities or thermal noise.
Fixing spin coordinates requires an integration constraint; see the spin
integration sections of :ref:`TSPIN <kw_ensemble_tspin>` and
:ref:`GLSD <kw_ensemble_glsd>`.

The grouped-response path derivative is computed from the converged DFT spins.
When :ref:`lambda_spin_response <kw_lambda_spin_response>` is used, the trainer
automatically sets that derived tangent to zero on excluded atom types.

To select only Fe for magnetic-force output and magnetic supervision, use::

  spin_dof_type Fe

Every selected type must also appear in ``spin_env_type``.
