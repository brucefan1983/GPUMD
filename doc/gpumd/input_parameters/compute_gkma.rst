.. _kw_compute_gkma:
.. index::
   single: compute_gkma (keyword in run.in)

:attr:`compute_gkma`
====================

The :attr:`compute_gkma` keyword can be used to calculate the modal heat current using the Green-Kubo modal analysis (:term:`GKMA`) method [Lv2016]_.
The results are written to the :ref:`heatmode.out output file <heatmode_out>`.


Syntax
------

.. code::

   compute_gkma sample_interval first_mode last_mode bin_option size

:attr:`sample_interval` is the sampling interval (in number of steps) used to compute the heat modal heat current.

:attr:`first_mode` and :attr:`last_mode` are the first and last mode, respectively, in the :ref:`eigenvector.in input file <eigenvector_in>` to include in the calculation.

:attr:`bin_option` determines which binning technique to use.
The options are ``bin_size`` and ``f_bin_size``.

:attr:`size` defines how the modes are added to each bin.
If :attr:`bin_option` is ``bin_size``, then this is an integer describing how many modes are included per bin.
If :attr:`bin_option` = ``f_bin_size``, then binning is by frequency and this is a float describing the bin size in THz.

Examples
--------

Example 1
^^^^^^^^^

.. code::

   compute_gkma 10 1 27216 f_bin_size 1.0

This means that

* you want to calculate the modal heat current with the :term:`GKMA` method
* the modal heat flux will be sampled every 10 steps
* the range of modes you want to include of calculations are from 1 to 27216
* you want to bin the modes by frequency with a bin size of 1 THz

Example 2
^^^^^^^^^

.. code::

   compute_gkma 10 1 27216 bin_size 1

This example is identical to Example 1, except the modes are binned by count.
Here, each bin only has one mode (i.e., all modes are included in the output).

Example 3
^^^^^^^^^

.. code::

   compute_gkma 10 1 27216 bin_size 10

This example is identical to Example 2, except each bin has 10 modes.

Caveats
-------

This computation can be very memory intensive.
The memory requirements are comparable to the size of the :ref:`eigenvector.in input file <eigenvector_in>`.

Depending on the number of steps to run, sampling interval, and number of bins, the :ref:`heatmode.out output file <heatmode_out>` can become very large as well (i.e., many GBs).

This keyword cannot be used in the same run as the :ref:`compute_hnema keyword <kw_compute_hnema>`.
The keyword that appears last will be used in the run.
