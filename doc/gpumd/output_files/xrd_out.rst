.. _xrd_out:
.. index::
   single: xrd.out (output file)

``xrd.out``
===========

:ref:`compute_xrd <kw_compute_xrd>` appends one block per sample. The first
line begins with ``# step`` and records the run-local step, number of
reciprocal points, number of sampled atoms, wavelength in Å, LP flag and
the three mesh spacings in Å⁻¹. A column header follows. When reading this
file, use the reported number of points to identify an incomplete block
if a run was interrupted while writing.

.. code-block:: none

   # h k l two_theta_deg intensity

Each data row is one reciprocal-space sample. The integers ``h k l``
index the sampling mesh, with vector components equal to the indices
times the corresponding spacings. They need not be Miller indices of
the crystal. The ``two_theta_deg`` column is the scattering angle
:math:`2\theta` in degrees and contains only angles strictly between
0° and 180°, even when ``two_theta`` uses these values as input bounds.

The ``intensity`` column is

.. math::

   I_{\mathrm{out}}(\mathbf{g})
   = \frac{L_P(\theta)\,|F(\mathbf{g})|^2}{N_{\mathrm{sel}}}.

Here :math:`N_{\mathrm{sel}}` is the number of atoms in the sampled group,
or the total number of atoms when ``group`` is omitted.
:ref:`compute_xrd <kw_compute_xrd>` defines :math:`F` and :math:`L_P`.
To form a powder pattern, sum these intensities in angle bins.
