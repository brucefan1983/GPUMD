.. _kw_compute_xrd:
.. index::
   single: compute_xrd (keyword in run.in)

:attr:`compute_xrd`
===================

Calculate kinematic X-ray diffraction intensities on a reciprocal-space grid.
For the reciprocal grid and kinematic diffraction convention, see
[Coleman2013]_.

Syntax
------

.. code-block:: none

   compute_xrd <interval> <wavelength> [options]

Parameters
----------

The sampling interval is a positive integer in MD steps and the wavelength is
in Å. Declare the command before each ``run`` that uses it. The first sample is
at step ``interval`` of that run.

* ``two_theta <min> <max>`` sets the scattering-angle range in degrees
  (default 1 to 179). The input must satisfy
  :math:`0^\circ \le (2\theta)_{\min} < (2\theta)_{\max} \le 180^\circ`.
  GPUMD omits the forward- and backscattering points at 0° and 180° for
  both settings of ``lp``.
* ``lp <0|1>`` disables or enables the Lorentz–polarization correction (default 1).
* ``group <method> <id>`` samples a selected atom group (default all atoms).
* ``scale <sx> <sy> <sz>`` multiplies the inverse box lengths by positive
  dimensionless factors (default 1 along each axis). A nonperiodic axis uses
  the mean inverse box length over the periodic axes.
* ``spacing <gx> <gy> <gz>`` sets fixed positive grid spacings in Å⁻¹.
  It is mutually exclusive with ``scale`` and required for a fully nonperiodic box.

Output
------

Write one intensity for each sampled point to :ref:`xrd.out <xrd_out>` in the working directory.
To obtain a powder pattern, sum the intensities in bins of equal scattering
angle.
See :ref:`xrd.out <xrd_out>` for the normalization of the stored intensity.

Examples
--------

For example, use ``compute_xrd 1000 1.5406 two_theta 10 100``.

See the `worked examples in GPUMD-Tutorials <https://github.com/brucefan1983/GPUMD-Tutorials>`_.

Notes
-----

Built-in form factors
^^^^^^^^^^^^^^^^^^^^^

GPUMD looks up the built-in neutral-atom X-ray form factor for each
sampled element using the exact symbol in ``model.xyz``. The table covers
98 elements, H through Cf (atomic numbers 1–98), using the plain element
labels in the source data (for example, ``Si`` for silicon).

The five-Gaussian fit is :math:`f_0(s)=c+\sum_{i=1}^{5}a_i\exp(-b_i s^2)`,
where :math:`s=\sin(\theta)/\lambda` in Å⁻¹ and
:math:`0 \le s \le 6` Å⁻¹. If the requested angle range extends past
this limit, adjust the wavelength or upper scattering-angle bound. Every
sampled element must be in the table. The form factors describe normal
elastic scattering by neutral atoms.

The coefficients use the fit in [Waasmaier1995]_.
The numbers are taken from the
`XrayDB data archive <https://github.com/xraypy/XrayDB/blob/96911505a105549100d183a6268cb3fdfd6445d3/data_sources/waasmaeir_kirfel.dat>`_,
whose `data license <https://github.com/xraypy/XrayDB/blob/96911505a105549100d183a6268cb3fdfd6445d3/LICENSE>`_
is CC0 1.0.

Structure factor and intensity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For mesh indices :math:`(h,k,l)` and spacings :math:`(g_x,g_y,g_z)`,
the reciprocal vector is :math:`\mathbf{g}=(h g_x,k g_y,l g_z)`
in cycles per Å. For wavelength :math:`\lambda` in Å, the Bragg angle and
the form-factor argument satisfy

.. math::

   2\theta = 2\arcsin\left(\frac{\lambda|\mathbf{g}|}{2}\right),
   \qquad s = \frac{\sin\theta}{\lambda} = \frac{|\mathbf{g}|}{2}.

Let :math:`\mathbf{r}_j` be the Cartesian position in Å of sampled atom
:math:`j`, :math:`f_j(s)` its atomic form factor and
:math:`N_{\mathrm{sel}}` the number of sampled atoms. The structure factor
and the uncorrected kinematic intensity are

.. math::

   F(\mathbf{g}) =
   \sum_{j=1}^{N_{\mathrm{sel}}} f_j(s)
   \exp\!\left(2\pi i\,\mathbf{g}\cdot\mathbf{r}_j\right),
   \qquad I_{\mathrm{kin}}(\mathbf{g}) = |F(\mathbf{g})|^2.

The multiplier for the Lorentz–polarization option is

.. math::

   L_P(\theta) =
   \begin{cases}
      \dfrac{1+\cos^2(2\theta)}{\cos\theta\,\sin^2\theta}, & \text{if lp is 1},\\
      1, & \text{if lp is 0}.
   \end{cases}

Sampling conventions
^^^^^^^^^^^^^^^^^^^^

Use an axis-aligned orthogonal box. Automated spacing is based on the whole
simulation box, including when a group is selected, and updates when the box
lengths change. Explicit spacing stays fixed. If the grid is incommensurate
with a periodic box, shifting an atom by a periodic box length can change
its phase. This can occur with user-specified spacing or noninteger scale.
The calculation uses the coordinate images supplied by the simulation.

For PIMD, RPMD and TRPMD, GPUMD samples bead 0, following the convention
of :ref:`compute_rdf <kw_compute_rdf>`.
