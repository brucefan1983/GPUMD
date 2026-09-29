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

The two structures of a pair must have the same number of atoms of each type.

A ``#`` starts a comment that extends to the end of the line.

Example
-------

::

  # two polymorphs of the same cell, target taken from train.xyz
  rutile anatase
  # a vacancy at two sites of one cell, with an explicit target and a higher weight
  vacancy_site1 vacancy_site2 0.42 2.0

Training and test pairs
-----------------------

A pair with both structures in :attr:`train.xyz` is a training pair, which enters the loss function and the column :attr:`rmse_ediff_train` of :ref:`loss.out <loss_out>`.
A pair with both structures in :attr:`test.xyz` is a test pair, which enters the column :attr:`rmse_ediff_test`.
A pair whose names occur in both files is both a training pair and a test pair.
A pair with a name that occurs in neither file, or with its two structures in different files, is skipped.
``nep`` prints the number of pairs of each kind and a warning with the number of skipped pairs.

Caveats
-------

* A line with fewer than two names or more than four fields, an invalid :attr:`ref_eV`, or a weight that is not a positive number is an input error.
* A pair of structures that differ in the number of atoms of some type is an input error.
* If a name occurs on more than one structure of a file, the pair refers to the first of them and a warning is printed.
