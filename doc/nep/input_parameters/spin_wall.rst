.. _kw_spin_wall:
.. index::
   single: spin_wall (keyword in nep.in)

``spin_wall``
=================

This optional ``model nep_spin`` keyword penalizes magnetic moment magnitudes
below a lower threshold or above an upper threshold. Both walls are off by
default for new training.

Shared parameters
-----------------

Give the lower wall first and the upper wall second. A numeric wall takes a
``threshold stiffness`` pair. One group applies to every element selected by
:ref:`spin_type <kw_spin_type>`.

.. code-block:: text

   spin_wall 2.5 1.5  5.5 2.0

Thresholds are in :math:`\mu_B`. Lower and upper stiffnesses are in
:math:`\mathrm{eV}/\mu_B` and :math:`\mathrm{eV}/\mu_B^3`, respectively.
Zero stiffness disables that side. To use only an upper wall, write
``spin_wall 0 0 5.5 2.0``. Enabled walls require positive thresholds. When
both sides are enabled, the lower threshold must be strictly below the upper
threshold.

Automatic thresholds
--------------------

In training, either side can use ``auto <margin> <stiffness>`` in place of its
numeric pair. The margin is non-negative and in :math:`\mu_B`.

.. code-block:: text

   spin_wall auto 0.2 1.5  auto 0.5 2.0

For each selected element, the lower threshold is the smallest moment
magnitude in ``train.xyz`` minus the lower margin. The upper threshold is the
largest magnitude plus the upper margin. For a training range of 1.5 to
3.0 :math:`\mu_B`, the example gives thresholds of 1.3 and 3.5 :math:`\mu_B`.
An automatic wall is disabled if the element has no training atoms or the
calculated threshold is nonpositive. The threshold order is checked after
resolving the automatic values.

Per-element parameters
----------------------

Give one group per element in ``type`` order. Each side can be numeric or
use ``auto``. For an element excluded by ``spin_type``, supply ``0 0 0 0``.
Use a single ``spin_wall`` line in ``nep.in``.

.. code-block:: text

   type 3 Fe Co O
   spin_type Fe Co
   spin_wall auto 0.2 1.5  3.5 2.0   0 0  auto 0.5 1.0   0 0 0 0

Fe has an automatic lower wall and a numeric upper wall. Co has only an
automatic upper wall. O has no walls.

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

When any wall is enabled, ``nep.txt`` stores one ``spin_wall`` line with four
numbers per element in ``type`` order, lower pair first. Disabled sides are
written as ``0 0``. MD uses the resolved values from the potential file.

Prediction uses the model values when ``spin_wall`` is omitted from
``nep.in``. A numeric ``spin_wall`` line overrides both sides. Use
``spin_wall 0 0 0 0`` to disable all walls for prediction.

Restart uses the resolved values saved in the model. An ``auto`` side with
positive stiffness reuses its saved values. Explicit numeric values and
disabled sides must agree with the saved model.
