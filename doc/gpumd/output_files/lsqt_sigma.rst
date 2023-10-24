.. _lsqt_sigma_out:
.. index::
   single: lsqt_sigma.out (output file)

``lsqt_sigma.out``
==================

This file contains the running electrical conductivity :math:`\Sigma(E,t)` as a function of energy :math:`E` and correlation time :math:`t`, from linear-scaling quantum transport (:term:`LSQT`) calculations.
It is produced when invoking the :ref:`compute_lsqt keyword <kw_compute_lsqt>` in the :ref:`run.in input file <run_in>`.

File format
-----------

* The number of columns equals the number of energy points. The energy values can be inferred from the parameters to the :ref:`compute_lsqt keyword <kw_compute_lsqt>`.

* The number of rows equals the product of the number of :term:`MD` steps for one run and the number of independent runs (supposed to be equilibrium ones), and the results from different runs can thus be averaged to reduce the statistical uncertainties.

* The electrical conductivity values are in units of S/m.