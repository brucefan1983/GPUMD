.. _kw_compute_elastic:
.. index::
   single: compute_elastic (keyword in run.in)

:attr:`compute_elastic`
=======================

This keyword computes static elastic constants from potential-energy changes under small deformations, relaxing the atomic positions at each fixed deformed cell.

Syntax
------

::

  compute_elastic <strain_value>

Parameters
----------

:attr:`strain_value` is the dimensionless amount of strain to be applied in the calculations and must lie in the interval :math:`(0, 0.1]`.

Output
------

.. _elastic_out:

The ``elastic.out`` file contains a comment header followed by a :math:`6\times6` matrix in units of GPa.
Both rows and columns use the Voigt order :math:`xx, yy, zz, yz, zx, xy`.
Thus, the entry in row :math:`a` and column :math:`b` is :math:`C_{ab}` in this order; for example, row 4, column 4 is :math:`C_{yzyz}`.
The file is overwritten each time this command is executed.
It can be read directly with NumPy::

   import numpy as np
   C = np.loadtxt("elastic.out")

Examples
--------

For example, the command::

  compute_elastic 0.01

means that one wants to compute the elastic constants with a strain of 0.01.

Notes
-----

This keyword must occur after the :ref:`potential keyword <kw_potential>`.
Use a relaxed reference cell with negligible residual stress, and check convergence with respect to the strain magnitude.
