.. _kw_spin_wall:
.. index::
   single: spin_wall_high (keyword in nep.in)
   single: spin_wall_low (keyword in nep.in)

:attr:`spin_wall_high` and :attr:`spin_wall_low`
================================================

These optional ``model nep_spin`` keywords penalize magnetic moment magnitudes
above an upper bound or below a lower bound. Both walls are off by default.

Shared parameters
-----------------

Use one ``threshold stiffness`` pair to apply the same wall to all spin-active
elements. Elements excluded by :ref:`spin_type <kw_spin_type>` are unaffected.

.. code-block:: text

   spin_wall_high 5.5 2.0
   spin_wall_low  2.5 1.5

Thresholds are in :math:`\mu_B`.
Upper and lower stiffnesses are in :math:`\mathrm{eV}/\mu_B^3` and
:math:`\mathrm{eV}/\mu_B`, respectively. Use a positive threshold when the
stiffness is nonzero. If both walls are enabled for an element, its lower
threshold must be strictly below its upper threshold.

Per-element parameters
----------------------

On each wall line, give one ``threshold stiffness`` pair for **every** element
in the order listed in ``type``, including elements without active spins.
Use ``0 0`` for an element that should have no wall on that side. For example:

.. code-block:: text

   type 3 Fe Co O
   spin_type Fe Co
   spin_wall_high 3.5 2.0  2.5 1.0  0 0
   spin_wall_low  1.5 1.5  0 0      0 0

The three pairs are for Fe, Co, and O, respectively. Fe has both walls, Co has
only an upper wall, and O has neither.

Automatic thresholds
--------------------

For new training, use ``auto <margin> <stiffness>`` to set thresholds from
``train.xyz``:

.. code-block:: text

   spin_wall_high auto 0.5 2.0
   spin_wall_low  auto 0.2 1.5

For each spin-active element, the upper threshold is its largest training
moment magnitude plus ``0.5``, and the lower threshold is its smallest
training moment magnitude minus ``0.2``. For example, a training range of
1.5 to 3.0 :math:`\mu_B` gives thresholds of 1.3 and 3.5 :math:`\mu_B`.
The margin and stiffness on each ``auto`` line are shared across elements;
the thresholds are calculated separately. The margin is non-negative and
in :math:`\mu_B`. Zero stiffness disables that wall. A type without training
atoms or with a nonpositive calculated threshold has that wall disabled.
Neither ``test.xyz`` nor ``response.xyz`` sets these bounds.

Choose one format per keyword. For example, the upper wall can use ``auto``
while the lower wall uses explicit per-element pairs. Do not repeat a
keyword to set individual elements.

Energy and magnetic force
-------------------------

For magnitude :math:`s = |\mathbf{S}|`, the energies are

.. math::

   E_\mathrm{high} = \frac{k_\mathrm{hi}}{3}(s-s_\mathrm{hi})^3
   \quad (s > s_\mathrm{hi}), \qquad
   E_\mathrm{low} = \frac{k_\mathrm{lo}s_\mathrm{lo}}{3}
   \left(1 - \frac{s^2}{s_\mathrm{lo}^2}\right)^3 \quad (s < s_\mathrm{lo}).

Each energy is zero outside its stated interval. The corresponding magnetic
forces are

.. math::

   \mathbf{F}_{S,\mathrm{high}} =
   \begin{cases}
   -k_\mathrm{hi}(s-s_\mathrm{hi})^2\dfrac{\mathbf{S}}{s}, & s>s_\mathrm{hi},\\
   \mathbf{0}, & \text{otherwise},
   \end{cases}

   \mathbf{F}_{S,\mathrm{low}} =
   \begin{cases}
   \dfrac{2k_\mathrm{lo}}{s_\mathrm{lo}}
   \left(1-\dfrac{s^2}{s_\mathrm{lo}^2}\right)^2\mathbf{S},
   & 0<s<s_\mathrm{lo},\\
   \mathbf{0}, & \text{otherwise}.
   \end{cases}

Here :math:`\mathbf{F}_S=-\partial E/\partial\mathbf{S}`. The total wall
energy and magnetic force are the sums of the enabled high and low terms.
Both forces are purely longitudinal. At exactly zero moment, the lower-wall
force is defined as zero. The walls do not directly contribute to atomic
forces, virials, or spin torque. A wall is a soft energy penalty, not a hard
bound or a thermostat.

Model files, prediction, and restart
------------------------------------

``nep.txt`` stores the numeric pairs for each enabled wall in element order,
including zero pairs for disabled types. This matches TorchNEP's spin-wall
metadata and does not change the parameter or scaler blocks. GPUMD MD uses
the pairs in the potential file. ``nep`` prediction also takes them from the
file by default, and a numeric pair in ``nep.in`` overrides the corresponding
wall for that prediction (use ``0 0`` to turn it off). Prediction cannot use
``auto``. Restart preserves the resolved values from ``nep.txt``; explicit
numeric restart values must agree with the saved values.
