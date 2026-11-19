.. _kw_compute_xrd:
.. index::
   single: compute_xrd (keyword in run.in)

:attr:`compute_xrd`
===================

Calculate kinematic X-ray diffraction intensities on a reciprocal-space grid.
Write one intensity for each sampled point to :ref:`xrd.txt <xrd_txt>`.
To obtain a powder pattern, sum the intensities in bins of equal scattering
angle. See the `worked examples in GPUMD-Tutorials <https://github.com/brucefan1983/GPUMD-Tutorials>`_.

Syntax
------

.. code-block:: none

   compute_xrd <interval> <wavelength> [options]

The sampling interval is a positive integer in MD steps and the wavelength is
in Å. Declare the command before each ``run`` that uses it. The first sample is
at step ``interval`` of that run.

* ``two_theta <min> <max>`` sets the scattering-angle range in degrees (default 1 to 179).
* ``lp <0|1>`` disables or enables the Lorentz–polarization correction (default 1).
* ``group <method> <id>`` samples a selected atom group (default all atoms).
* ``scale <sx> <sy> <sz>`` multiplies the inverse box lengths by positive
  dimensionless factors (default 1 along each axis). A nonperiodic axis uses
  the mean inverse box length over the periodic axes.
* ``spacing <gx> <gy> <gz>`` sets fixed positive grid spacings in Å⁻¹.
  It is mutually exclusive with ``scale`` and required for a fully nonperiodic box.
* ``file <filename>`` sets the output filename (default ``xrd.txt``).
* ``form_factor <element> <a1> <b1> <a2> <b2> <a3> <b3> <a4> <b4> <c>``
  supplies one set of coefficients per sampled element. All coefficients must
  be finite. GPUMD supplies no built-in scattering table. Use coefficients
  appropriate for the elements and X-ray scattering model in your calculation.

The atomic form factor is
:math:`f(s)=c+\sum_{i=1}^{4}a_i\exp(-b_i s^2)`, where
:math:`s=\sin(\theta)/\lambda` in Å⁻¹. For an analytic check with unit
scatterers, use four zero amplitude–exponent pairs and set ``c`` to 1.

Sampling conventions
--------------------

Use an axis-aligned orthogonal box. Automated spacing is based on the whole
simulation box, including when a group is selected, and updates when the box
lengths change. Explicit spacing stays fixed. If the grid is incommensurate
with a periodic box, shifting an atom by a periodic box length can change
its phase. This can occur with user-specified spacing or noninteger scale.
The calculation uses the coordinate images supplied by the simulation.

For PIMD, RPMD and TRPMD, GPUMD samples bead 0, following the convention
of :ref:`compute_rdf <kw_compute_rdf>`.
