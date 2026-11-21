.. _xrd_txt:
.. index::
   single: xrd.txt (output file)

``xrd.txt``
===========

:ref:`compute_xrd <kw_compute_xrd>` appends one block per sample. The first
line begins with ``# step`` and records the run-local step, number of
reciprocal points, number of sampled atoms, wavelength in Å, LP flag and
the three mesh spacings in Å⁻¹. A column header follows.

.. code-block:: none

   # h k l two_theta_deg intensity

Each data row is one reciprocal-space sample. The integers ``h k l``
index the sampling mesh, with vector components equal to the indices
times the corresponding spacings. They need not be Miller indices of
the crystal. The angle is in degrees. Intensity is the squared modulus
of the structure factor per sampled atom, with the Lorentz–polarization
factor applied if enabled. Angle bins can be formed by summing these rows.
