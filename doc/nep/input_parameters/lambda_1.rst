.. _kw_lambda_1:
.. index::
   single: lambda_1 (keyword in nep.in)

:attr:`lambda_1`
================

This keyword sets the weight :math:`\lambda_1` of the :math:`\mathcal{L}_1`-norm regularization term in the :ref:`loss function <nep_loss_function>`.

Syntax
------

::

  lambda_1 <weight>

Parameters
----------

Here, :attr:`<weight>` represents :math:`\lambda_1`, which can be set to any non-negative values.
The default value is :math:`\lambda_1 = \sqrt{N_\mathrm{par}/N_\mathrm{typ}}/1000`, where :math:`N_\mathrm{par}` is the total number of training parameters and :math:`N_\mathrm{typ}` is the number of atom types.

Output
------

None.

Examples
--------

::

  lambda_1 0.05

Notes
-----

None.
