.. _kw_lambda_d:
.. index::
   single: lambda_d (keyword in nep.in)

:attr:`lambda_d`
================

This keyword sets the weight :math:`\lambda_d` of the loss term associated with **energy differences** between structures in the :ref:`loss function <nep_loss_function>`.
The syntax is::

  lambda_d <weight>

Here, :attr:`<weight>` represents :math:`\lambda_d`, which must satisfy :math:`\lambda_d \geq 0` and defaults to :math:`\lambda_d = 0`.
The term is thus inactive unless the keyword sets a positive weight.

Each line of the file :ref:`ediff.in <ediff_in>` defines a linear combination :math:`k` of the total energies of named structures, such as the difference :math:`E_a - E_b` or the formation energy :math:`E_\mathrm{vac} + \frac{1}{2} E_\mathrm{O_2} - E_\mathrm{bulk}`.
The contribution to the loss is

.. math::

   \lambda_d \left( \frac{1}{N_\mathrm{comb}} \sum_k w_k
   \left[ \sum_i c_{ki} \left( E_i - E_i^\mathrm{tar} \right) \right]^2 \right)^{1/2},

where :math:`E_i` and :math:`E_i^\mathrm{tar}` are the predicted and target total energies of structure :math:`i` in eV, :math:`c_{ki}` is its coefficient in combination :math:`k`, and :math:`w_k` is the weight of the combination.
The term is thus in eV of total energy, while the energy term of the loss function is in eV/atom.
The sum runs over the :math:`N_\mathrm{comb}` combinations whose structures all lie in the current mini-batch.
The fields :attr:`weight` and :attr:`energy_weight` of the structures in :attr:`train.xyz` do not enter the term.

A positive :math:`\lambda_d` requires two inputs:

1. a :attr:`name=<label>` field on the comment line of each structure that takes part in a combination, see :ref:`train.xyz and test.xyz <train_test_xyz>`,
2. the file :ref:`ediff.in <ediff_in>`, which lists the combinations.

Each combination must be balanced in the number of atoms of each type, that is :math:`\sum_i c_{ki} n_i(t) = 0` for every type :math:`t`, where :math:`n_i(t)` is the number of atoms of type :math:`t` in structure :math:`i`.
Any uniform or per-type offset of the predicted energies then cancels in the combination.

``nep`` stops with an input error if :attr:`ediff.in` is missing or if none of its combinations has all its structures in :attr:`train.xyz`.
With :math:`\lambda_d = 0`, an :attr:`ediff.in` file is ignored.
The keyword is only available for potential models and is an input error together with :attr:`model_type 1` or :attr:`model_type 2`, also in prediction mode.
In prediction mode of a potential model, the keyword has no effect.

When the term is active, :ref:`loss.out <loss_out>` has two additional columns, :attr:`rmse_ediff_train` and :attr:`rmse_ediff_test`.

Mini-batches
------------

With a :attr:`batch` size smaller than the number of training structures, ``nep`` keeps the structures of each training combination in one mini-batch.
The mini-batches then differ from those of a run without the term, also for the structures that take part in no combination.
Structures linked through combinations form a group.
The groups are dealt one by one into the mini-batch with the fewest structures, the larger groups first and groups of equal size in the order of their mean energy per atom.
Large groups can make a mini-batch larger than the batch size, and ``nep`` then prints a warning with the size of the largest mini-batch.
A mini-batch left empty is dropped, and ``nep`` prints the reduced number of mini-batches.
