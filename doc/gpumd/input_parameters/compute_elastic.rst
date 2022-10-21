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

  compute_elastic strain_value symmetry_type

:attr:`strain_value` is the amount of strain to be applied in the calculations.

:attr:`symmetry_type` is the symmetry type of the material considered.
Currently, it can only be :attr:`cubic`.

Example
-------
For example, the command::

  compute_elastic 0.01 cubic

means that one wants to compute the elastic constants for a cubic system with a strain of 0.01.

Caveats
-------
This keyword must occur after the :ref:`potential keyword <kw_potential>`.
