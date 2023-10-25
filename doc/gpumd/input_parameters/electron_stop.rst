.. _kw_electron_stop:
.. index::
   single: electron_stop (keyword in run.in)

:attr:`electron_stop`
=====================

This keyword is used to apply electron stopping forces to high-energy atoms during a run. This is usually used in radiation damage simulations.

Syntax
------

This keyword is used as follows::

  electron_stop <file>

Here, :attr:`file` is the file containing a table of the stopping power data.
The first line of the file should have 3 values: the first is the number of data points to be listed :math:`N`; the second is the minimum of the energy range :math:`E_{\rm min}`; the third is the maximum of the energy range :math:`E_{\rm max}`. The stopping power data listed after this line should have :math:`N` lines, each corresponding to one energy, which increases linearly from :math:`E_{\rm min}` to :math:`E_{\rm max}` with an interval of :math:`(E_{\rm max} - E_{\rm min})/(N-1)`. For these :math:`N` lines, the number of columns is the number of species for the used potential. That is, each species should have a column of stopping power data in this file. The order of the species here should follow that as defined in the potential file.

Example
-------

An example is::

   electron_stop my_stopping_power.txt