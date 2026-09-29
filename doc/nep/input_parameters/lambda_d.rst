.. _kw_lambda_d:
.. index::
   single: lambda_d (keyword in nep.in)

:attr:`lambda_d`
================

This keyword sets the weight :math:`\lambda_d` of the loss term associated with **energy differences** between pairs of structures in the :ref:`loss function <nep_loss_function>`.
The syntax is::

  lambda_d <weight>

Here, :attr:`<weight>` represents :math:`\lambda_d`, which must satisfy :math:`\lambda_d \geq 0` and defaults to :math:`\lambda_d = 0`.
The term is thus inactive unless the keyword sets a positive weight.

The contribution to the loss is

.. math::

   \lambda_d \left( \frac{1}{N_\mathrm{pairs}} \sum_{(a,b)} w_{ab}
   \left[ \left( E_a - E_b \right) - \Delta E_{ab}^\mathrm{tar} \right]^2 \right)^{1/2},

where :math:`E_a` and :math:`E_b` are the predicted total energies of the two structures of a pair in eV, :math:`\Delta E_{ab}^\mathrm{tar}` is the target energy difference, and :math:`w_{ab}` is the weight of the pair.
The sum runs over the :math:`N_\mathrm{pairs}` pairs whose two structures both lie in the current mini-batch.

A positive :math:`\lambda_d` requires two inputs:

1. a :attr:`name=<label>` field on the comment line of each structure that takes part in a pair, see :ref:`train.xyz and test.xyz <train_test_xyz>`,
2. the file :ref:`ediff.in <ediff_in>`, which lists the pairs.

The two structures of a pair must have the same number of atoms of each type.
Any uniform or per-type offset of the predicted energies then cancels in their difference.

``nep`` stops with an input error if :attr:`ediff.in` is missing or if none of its pairs has both structures in :attr:`train.xyz`.
With :math:`\lambda_d = 0`, an :attr:`ediff.in` file is ignored.
The keyword is only available for potential models and is an input error together with :attr:`model_type 1` or :attr:`model_type 2`.

When the term is active, :ref:`loss.out <loss_out>` has two additional columns, :attr:`rmse_ediff_train` and :attr:`rmse_ediff_test`.

Caveats
-------

* A training pair enters the loss only in the generations whose mini-batch holds both of its structures.
  With a batch size of at least the number of training structures, every pair enters the loss in every generation.
  ``nep`` prints a warning with the number of pairs that span two mini-batches.
