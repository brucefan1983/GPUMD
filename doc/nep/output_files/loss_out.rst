.. _loss_out:
.. index::
   single: loss.txt (output file)

``loss.out``
============

This files contains the terms that enter the :ref:`loss function <nep_loss_function>` for every 100-th generation.

Each row contains the following fields::

  gen L_t L_1 L_2 L_e_train L_f_train L_v_train L_e_test L_f_test L_v_test

where

* :attr:`gen` is the current generation.
* :attr:`L_t` is the total loss function.
* :attr:`L_1` is the loss function related to the :math:`\mathcal{L}_1` regularization.
* :attr:`L_2` is the loss function related to the :math:`\mathcal{L}_2` regularization.
* :attr:`L_e_train` is the energy RMSE (in units of eV/atom) for the training set.
* :attr:`L_f_train` is the force RMSE (in units of eV/Å) for the training set.
* :attr:`L_v_train` is the virial RMSE (in units of eV/atom) for the training set.
* :attr:`L_e_test` is the energy RMSE (in units of eV/atom) for the test set.
* :attr:`L_f_test` is the force RMSE (in units of eV/Å) for the test set.
* :attr:`L_v_test` is the virial RMSE (in units of eV/atom) for the test set.
