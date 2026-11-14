.. _spin_restart:

``restart.xyz`` (spin)
======================

Follow the :ref:`restart.xyz instructions <restart_xyz>` to continue a NEP-spin simulation.
The restart file additionally contains the per-atom magnetic moments as ``spin:R:3``.
When TSPIN spin velocities are initialized, it also contains ``spin_vel:R:3``.
TSPIN reuses these spin velocities; if they are absent, it initializes them from the input temperature and seed.
SIB and GLSD have no independent spin velocity.
