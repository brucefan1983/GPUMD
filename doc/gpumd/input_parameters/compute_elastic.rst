.. _kw_compute_elastic:
.. index::
   single: compute_elastic (keyword in run.in)

:attr:`compute_elastic`
=======================

This keyword is used to compute the elastic constants.
The results are written to the file ``elastic.out``.

Syntax
------

This keyword is used as follows::

  compute_elastic <strain_value> 

:attr:`strain_value` is the amount of strain to be applied in the calculations.


Example
-------
For example, the command::

  compute_elastic 0.001 

means that one wants to compute the elastic constants with a strain of 0.001.

Caveats
-------
This keyword must occur after the :ref:`potential keyword <kw_potential>`.
