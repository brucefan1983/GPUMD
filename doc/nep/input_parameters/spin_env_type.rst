.. _kw_spin_env_type:
.. index::
   single: spin_env_type (keyword in nep.in)

:attr:`spin_env_type`
======================

This keyword selects atom types allowed to contribute as magnetic neighbors::

  spin_env_type <type_1> [<type_2> ...]

Names must be unique and must appear in :ref:`type <kw_type>`. If this keyword
is absent, its mask is copied from
:ref:`spin_dof_type <kw_spin_dof_type>`; it does not independently default to
all atom types.

Every type selected by ``spin_dof_type`` must also be an environment type, so
``spin_dof_type`` must be a subset of ``spin_env_type``. The two masks may
differ when a type should affect local magnetic descriptors but should not
receive a public magnetic force. For example::

  spin_dof_type Fe
  spin_env_type Fe Ge

Here Fe receives magnetic-force output and magnetic supervision, while both
Fe and Ge atoms may enter the spin neighbor environment. These masks do not
freeze Ge spin coordinates during MD; see :ref:`spin_dof_type <kw_spin_dof_type>`.
