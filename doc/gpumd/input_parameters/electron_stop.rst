.. _kw_electron_stop:
.. index::
   single: electron_stop (keyword in run.in)

:attr:`electron_stop`
=====================

This keyword is used to apply electron stopping forces to high-energy atoms during a run. 
This is usually used in radiation damage simulations.

Syntax
------

This keyword is used as follows::

  electron_stop <file>

The first line of the file should have 3 values:
* the first is the number of data points to be listed :math:`N`;
* the second is the minimum of the energy range :math:`E_{\rm min}`;
* the third is the maximum of the energy range :math:`E_{\rm max}`.

The stopping power data listed after this line should have :math:`N` lines, each corresponding to one energy, which increases linearly from :math:`E_{\rm min}` to :math:`E_{\rm max}` with a spacing of :math:`(E_{\rm max} - E_{\rm min})/(N-1)`.
For these :math:`N` lines, the number of columns is the number of species for the potential energy model used.
That is, there should be one column with the stopping power for each species.
The order of the species should follow that as defined in the potential file.

Example
-------

An example is::

   electron_stop my_stopping_power.txt