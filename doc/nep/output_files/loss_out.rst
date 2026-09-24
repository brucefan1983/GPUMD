.. _loss_out:
.. index::
   single: loss.txt (output file)

``loss.out``
============

This files contains the terms that enter the :ref:`loss function <nep_loss_function>`, written every :ref:`output_interval <kw_output_interval>` generations (100 by default).

File format
-----------

The file starts with a header of lines beginning with ``#``::

  # format_version 1
  # output_interval 100
  # columns generation total L1 L2 rmse_energy_train ...

* :attr:`format_version` is the version of the file format.
* :attr:`output_interval` is the number of generations between two rows.
* :attr:`columns` lists the names of the columns in each row.

Each run writes this header before its rows.
A file that a restarted run appends to thus contains several headers.

If a potential model is trained, the columns are::

  generation total L1 L2 rmse_energy_train rmse_force_train rmse_virial_train rmse_energy_test rmse_force_test rmse_virial_test

where

* :attr:`generation` is the current generation.
* :attr:`total` is the total loss function.
* :attr:`L1` is the loss function related to the :math:`\mathcal{L}_1` regularization.
* :attr:`L2` is the loss function related to the :math:`\mathcal{L}_2` regularization.
* :attr:`rmse_energy_train` is the energy RMSE (in units of eV/atom) for the training set.
* :attr:`rmse_force_train` is the force RMSE (in units of eV/Å) for the training set.
* :attr:`rmse_virial_train` is the virial RMSE (in units of eV/atom) for the training set.
* :attr:`rmse_energy_test` is the energy RMSE (in units of eV/atom) for the test set.
* :attr:`rmse_force_test` is the force RMSE (in units of eV/Å) for the test set.
* :attr:`rmse_virial_test` is the virial RMSE (in units of eV/atom) for the test set.

If a potential model with charges (:ref:`charge_mode <kw_charge_mode>` or ``charge_vdw``) is trained, the columns are::

  generation total L1 L2 rmse_energy_train rmse_force_train rmse_virial_train rmse_charge_train rmse_bec_train rmse_energy_test rmse_force_test rmse_virial_test rmse_charge_test rmse_bec_test

where

* :attr:`rmse_charge_train` and :attr:`rmse_charge_test` are the RMSE of the total charge (in units of e/atom) for the training and test sets.
* :attr:`rmse_bec_train` and :attr:`rmse_bec_test` are the :term:`BEC` RMSE (in units of e) for the training and test sets.

If a dipole model is trained, the columns are::

  generation total L1 L2 rmse_dipole_train rmse_dipole_test

where

* :attr:`rmse_dipole_train` is the dipole RMSE (per atom) for the training set.
* :attr:`rmse_dipole_test` is the dipole RMSE (per atom) for the test set.

If a polarizability model is trained, the columns are::

  generation total L1 L2 rmse_polarizability_train rmse_polarizability_test

where

* :attr:`rmse_polarizability_train` is the polarizability RMSE (per atom) for the training set.
* :attr:`rmse_polarizability_test` is the polarizability RMSE (per atom) for the test set.

GNEP
----

The ``gnep`` executable writes one row per epoch.
Its header contains the lines ``# format_version 1`` and::

  # columns epoch total rmse_energy_train rmse_force_train rmse_virial_train rmse_energy_test rmse_force_test rmse_virial_test learning_rate time

where

* :attr:`epoch` is the current epoch.
* :attr:`total` is the total loss function.
* :attr:`rmse_*` are the RMSE values defined for a potential model.
* :attr:`learning_rate` is the current learning rate.
* :attr:`time` is the accumulated wall time of the training epochs (in units of s).
