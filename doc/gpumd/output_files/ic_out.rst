.. _ic_out:
.. index::
   single: ic.out (output file)

``ic.out``
===========

This file contains the Nernst--Einstein estimate of the ionic conductivity (:term:`IC`) based on the self mean-square displacement (:term:`MSD`).
It is generated when invoking the :ref:`compute_ic keyword <kw_compute_ic>`.
The definition and the omission of correlations between different ions are explained on that keyword's page.

File format
-----------
The data in this file are organized as follows:

* column 1: correlation time (in units of ps)
* column 2: IC (in units of mS/cm) in the :math:`x` direction
* column 3: IC (in units of mS/cm) in the :math:`y` direction
* column 4: IC (in units of mS/cm) in the :math:`z` direction

Only atoms of the type selected by the :attr:`type_iex` argument of :ref:`compute_ic <kw_compute_ic>` contribute to this output.
