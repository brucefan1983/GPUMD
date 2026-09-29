.. _ediff_in:
.. index::
   single: ediff.in (input file)

``ediff.in``
============

This file lists pairs of structures whose **energy difference** enters the loss function through the keyword :ref:`lambda_d <kw_lambda_d>`.
It is read only when :math:`\lambda_d > 0`.

File format
-----------

Each line holds one pair::

  <name_a> <name_b> [<ref_eV>] [<weight>]

* :attr:`name_a` and :attr:`name_b` are the labels given by the :attr:`name=<label>` fields on the comment lines of the structures in :ref:`train.xyz and test.xyz <train_test_xyz>`.
  Names are matched case-insensitively.
* :attr:`ref_eV` is optional and gives the target energy difference :math:`E_a - E_b` as a difference of total energies in eV.
  If it is omitted, the target is the difference of the target total energies of the two structures.
* :attr:`weight` is optional, must be positive, and defaults to 1.
  It scales the contribution of the pair to the loss.

A ``#`` starts a comment that extends to the end of the line.

Example
-------

::

  # vacancy formation, target taken from train.xyz
  vacancy perfect
  # an interstitial with an explicit target and a higher weight
  interstitial perfect 3.42 2.0

Training and test pairs
-----------------------

A pair with both structures in :attr:`train.xyz` is a training pair, which enters the loss function and the column :attr:`rmse_ediff_train` of :ref:`loss.out <loss_out>`.
A pair with both structures in :attr:`test.xyz` is a test pair, which enters the column :attr:`rmse_ediff_test`.
A pair whose names occur in both files is both a training pair and a test pair.
A pair with a name that occurs in neither file, or with its two structures in different files, is skipped with a warning.

Caveats
-------

* If a name occurs on more than one structure of a file, the pair refers to the first of them and a warning is printed.
* A line with fewer than two names or with an invalid :attr:`ref_eV` is skipped with a warning.
  An invalid :attr:`weight` is replaced by 1 with a warning.
