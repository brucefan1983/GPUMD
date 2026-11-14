.. _adf_out:
.. index::
   single: adf.out (output file)

``adf.out``
===========

This file contains the Angular Distribution Function (:term:`ADF`). 
It is generated when the :ref:`compute_adf keyword <kw_compute_adf>` is used.

File format
-----------
For global ADF, the data in this file are organized as follows:

- Column 1: Angular bin centers (in degrees)
- Column 2: The (:term:`ADF`) for the entire system

For local ADF, the data are organized as follows:

- Column 1: Angular bin centers (in degrees)
- The next Ntriples columns: The (:term:`ADF`) for each specific atom triple


Each sampling step appends a header followed by ``num_bins`` rows.
The ``step`` value in the header is the zero-based MD step index within the run.
The local columns follow the order of triples in the input command.

The angular bins have width :math:`\Delta\theta=180^\circ/\mathtt{num\_bins}` and centers :math:`(b+1/2)\Delta\theta`, where :math:`b=0,\ldots,\mathtt{num\_bins}-1`.
The bins include their lower edge and exclude their upper edge, except that :math:`180^\circ` is included in the last bin.
The ADF values are probability densities in degrees\ :sup:`-1`.
Each block uses all samples accumulated since the start of the current run, with the normalization described for :ref:`compute_adf <kw_compute_adf>`.
A column with no valid angles is all zeros.
Each new run starts a fresh histogram and appends its output to the file.
