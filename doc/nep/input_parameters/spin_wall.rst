.. _kw_spin_wall:
.. index::
   single: spin_wall_high (keyword in nep.in)
   single: spin_wall_low (keyword in nep.in)

:attr:`spin_wall_high` and :attr:`spin_wall_low`
================================================

These optional ``model nep_spin`` keywords penalize magnetic moment magnitudes
above an upper bound or below a lower bound. Both walls are off by default.

.. code-block:: text

   spin_wall_high 5.5 2.0
   spin_wall_low  2.5 1.5

Supply one ``threshold stiffness`` pair for all spin-active elements, or one
pair per element in the ``type`` order. Thresholds are in :math:`\mu_B`.
Upper and lower stiffnesses are in :math:`\mathrm{eV}/\mu_B^3` and
:math:`\mathrm{eV}/\mu_B`, respectively. Zero stiffness disables a wall for
that element. An enabled wall requires a positive threshold and an active
:ref:`spin_type <kw_spin_type>`. When both walls are enabled for an element,
its lower threshold must be strictly below its upper threshold.

For new training, either keyword also accepts ``auto <margin> <stiffness>``.
For each active element, the upper threshold is its maximum moment magnitude
in ``train.xyz`` plus the margin, and the lower threshold is its minimum
minus the margin. The margin is non-negative and in :math:`\mu_B`. A type
without training atoms or with a nonpositive resulting threshold has that
wall disabled. Neither ``test.xyz`` nor ``response.xyz`` sets these bounds.

For magnitude :math:`s = |\mathbf{S}|`, the energies are

.. math::

   E_\mathrm{high} = \frac{k_\mathrm{hi}}{3}(s-s_\mathrm{hi})^3
   \quad (s > s_\mathrm{hi}), \qquad
   E_\mathrm{low} = \frac{k_\mathrm{lo}s_\mathrm{lo}}{3}
   \left(1 - \frac{s^2}{s_\mathrm{lo}^2}\right)^3 \quad (s < s_\mathrm{lo}).

Each energy is zero outside its stated interval. The magnetic force is
:math:`-\partial E/\partial\mathbf{S}` and is purely longitudinal. It is zero
at exactly zero moment, even with a lower wall. The walls do not directly
contribute to atomic forces, virials, or spin torque. A wall is a soft energy
penalty, not a hard bound or a thermostat.

``nep.txt`` stores the numeric pairs for each enabled wall in element order,
including zero pairs for disabled types. This matches TorchNEP's spin-wall
metadata and does not change the parameter or scaler blocks. GPUMD MD uses
the pairs in the potential file. ``nep`` prediction also takes them from the
file by default, and a numeric pair in ``nep.in`` overrides the corresponding
wall for that prediction (use ``0 0`` to turn it off). Prediction cannot use
``auto``. Restart preserves the resolved values from ``nep.txt``; explicit
numeric restart values must agree with the saved values.
