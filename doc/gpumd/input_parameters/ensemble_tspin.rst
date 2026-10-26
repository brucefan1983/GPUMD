.. _kw_ensemble_tspin:
.. _kw_nvt_tspin:
.. _kw_npt_tspin:
.. index::
   single: nvt_tspin (keyword in run.in)
   single: npt_tspin (keyword in run.in)
   single: TSPIN integrator

:attr:`ensemble` (TSPIN)
========================

Use TSPIN to propagate the Cartesian spin coordinates of a Spin potential.
Choose ``nvt_tspin`` for a fixed simulation cell, optionally with lattice
dynamics, or ``npt_tspin`` when the simulation cell and pressure must also be
controlled.

The implementation is compositional. Both commands add the same spin
Nose--Hoover-chain integrator. ``nvt_tspin`` combines it with the standard
lattice Nose--Hoover chain, whereas ``npt_tspin`` combines it with the
:ref:`MTTK thermostat and barostat <kw_ensemble_mttk>`. TSPIN does not contain
a second implementation of the lattice thermostat or barostat.

Syntax
------

For fixed-cell NVT dynamics, use::

    ensemble nvt_tspin <T_1> <T_2> <T_coup> \
             [lattice on|off] \
             [mass_factor <value> | mass_factor <element> <value> ...] \
             [seed <value>]

For pressure-controlled NPT dynamics, use::

    ensemble npt_tspin temp <T_1> <T_2> <pressure_terms> \
             [tperiod <tau_temp>] [pperiod <tau_press>] \
             [mass_factor <value> | mass_factor <element> <value> ...] \
             [seed <value>]

The ``npt_tspin`` options through ``pperiod`` have the same syntax, meaning,
and defaults as ``npt_mttk``. The pressure terms can be ``iso``, ``aniso``,
``tri``, or a supported combination of ``x``, ``y``, ``z``, ``xy``, ``xz``,
and ``yz``. The optional TSPIN pairs must come after all MTTK options.

Examples
--------

Run fixed-cell spin--lattice dynamics at 300 K::

    ensemble nvt_tspin 300 300 100 mass_factor 0.001 seed 12345

Keep atomic positions and lattice velocities fixed and propagate only the
spin coordinates::

    ensemble nvt_tspin 300 300 100 lattice off \
             mass_factor 0.001 seed 12345

For a Fe--Ge structure, propagate Fe spins with a coefficient of ``0.001``
and keep Ge spins fixed::

    ensemble nvt_tspin 300 300 100 \
             mass_factor Fe 0.001 Ge 0 seed 12345

Run isotropic pressure-controlled spin--lattice dynamics at 300 K and 0 GPa::

    ensemble npt_tspin temp 300 300 iso 0 0 \
             tperiod 100 pperiod 1000 mass_factor 0.001 seed 12345

In both examples, ``mass_factor`` is a spin-mass coefficient. It does not
replace or change the lattice atomic masses.

Parameters
----------

``T_1``, ``T_2``
  Initial and final target temperatures in K. The target is ramped linearly
  during each ``run`` and is shared by the lattice and spin thermostats.

``T_coup``
  Thermostat coupling period in timesteps for ``nvt_tspin``. It controls both
  lattice and spin thermostats with ``lattice on``, and only the spin
  thermostat with ``lattice off``. It must be finite and at least ``1``.

``lattice on|off``
  Available only for ``nvt_tspin`` and defaults to ``on``. With ``on``, the
  lattice positions and velocities are integrated and thermostatted by the
  standard Nose--Hoover chain. With ``off``, GPUMD leaves every atomic
  position and lattice velocity unchanged while continuing to evaluate the
  Spin potential and integrate the spin coordinates. Supplied lattice
  velocities are preserved rather than cleared.

``mass_factor <value>``
  A single positive finite value applies to every element. The code default is
  ``1.0``. A practical starting value for magnetic-dynamics runs is ``0.001``.
  A smaller positive value produces a faster spin response and can require a
  smaller timestep, so check trajectory and temperature stability for the
  system being studied.

``mass_factor <element> <value> ...``
  Sets the coefficient by element name. Every element present in
  ``model.xyz`` must be listed exactly once. Values must be finite and
  non-negative, and at least one element must have a positive value. A value
  of ``0`` is a TSPIN freeze marker: GPUMD excludes those spins from the spin
  thermostat degrees of freedom, sets their spin velocities to zero, and does
  not update their spin coordinates. Zero is never used as a physical mass or
  as a divisor.

``seed <value>``
  Positive integer used to generate spin velocities when ``spin_vel:R:3`` is
  absent. The default is ``12345``. The seed is ignored when initialized spin
  velocities are supplied.

Spin integration and ``spin_dof_type``
--------------------------------------

TSPIN integrates every ``spin:R:3`` coordinate in ``model.xyz`` and includes
all of them in the spin thermostat degrees of freedom. It consumes the public
``mforce:R:3`` returned by the potential, but it does not read
``spin_dof_type``.

``spin_dof_type`` is a Spin-potential response mask. For an excluded atom
type, the potential returns zero public magnetic force. This does not freeze
the spin, clear its spin velocity, or remove it from TSPIN. With zero magnetic
force, that coordinate can still move because of its current spin velocity
and the spin thermostat. Fixing selected spin coordinates therefore requires
an explicit integration constraint; ``spin_dof_type`` must not be used for
that purpose.

State initialization and restart
--------------------------------

Each atom must have a ``spin:R:3`` vector. TSPIN propagates the full Cartesian
vector and does not normalize its length.

If ``spin_vel:R:3`` is absent, GPUMD generates spin velocities from ``seed``
and rescales them separately for each atom type to ``T_1``. If
``spin_vel:R:3`` is present, GPUMD uses every supplied value and ignores
``seed``. For an element whose mass factor is zero, any supplied spin velocity
is replaced by zero. Initialized spin velocities are reused by later ``run``
commands.
They are also written by ``dump_restart`` and reused when its ``restart.xyz``
is loaded as ``model.xyz``.

Pressure coupling in ``npt_tspin``
----------------------------------

The MTTK barostat acts on the simulation cell, atomic positions, and lattice
momenta. It neither rescales nor rotates the Cartesian spin vectors. The
instantaneous pressure contains the lattice kinetic term and the full
potential virial of :math:`U(\boldsymbol R,\boldsymbol S)`, but not the TSPIN
kinetic energy. Spin affects the cell through the Spin-potential virial.
Because pressure control requires cell and lattice dynamics, ``npt_tspin``
does not accept ``lattice off``.

Output and checks
-----------------

Write the propagated state and magnetic force with::

    dump_xyz 100 spin.xyz spin mforce spin_velocity

The extended-XYZ fields are ``spin:R:3``, ``mforce:R:3``, and
``spin_vel:R:3``. Check that the spin temperature remains stable for the
chosen ``mass_factor`` and timestep. With ``lattice on``, also check the
lattice temperature. For ``npt_tspin``, check cell and pressure stability.

The run log reports either ``integrate the lattice with a Nose-Hoover chain``
or ``keep lattice positions and velocities fixed``. When using ``lattice
off``, include ``velocity`` in a short ``dump_xyz`` run; positions are written
automatically. Confirm that both remain unchanged while ``spin`` evolves.
The log also prints the resolved spin mass factor for each element. For an
element assigned ``0``, confirm that both ``spin`` and ``spin_velocity`` stay
unchanged while an element with a positive factor evolves.

Restrictions
------------

TSPIN requires exactly one Spin potential, a positive timestep, positive
target temperatures, and finite integration parameters. ``npt_tspin`` is the
only pressure-controlled TSPIN variant; ``npt_ber_tspin`` and
``npt_scr_tspin`` are not implemented. The MTTK restriction
``pperiod >= 200`` also applies to ``npt_tspin``. The ``lattice`` keyword is
only valid for ``nvt_tspin`` and accepts only ``on`` or ``off``. Element-wise
``mass_factor`` input must cover every element present in ``model.xyz``;
unknown, missing, or repeated element names are rejected.

Defaults
--------

``lattice`` defaults to ``on``, ``mass_factor`` defaults to ``1.0``, and
``seed`` defaults to ``12345``.
For ``npt_tspin``, MTTK retains its own defaults, including ``tperiod 100``
and ``pperiod 1000``.
