.. _ediff_in:
.. index::
   single: ediff.in (input file)

``ediff.in``
============

This file lists linear combinations of the total energies of structures, such as energy differences and formation energies, that enter the loss function through the keyword :ref:`lambda_d <kw_lambda_d>`.
It is read only when :math:`\lambda_d > 0`.

File format
-----------

Each line holds one combination::

  [+|-] <term> {+|- <term>} [<weight>]

* A :attr:`term` is either :attr:`<name>` or :attr:`<coefficient>*<name>`, without spaces.
  :attr:`name` is the label given by the :attr:`name=<label>` field on the comment line of a structure in :ref:`train.xyz and test.xyz <train_test_xyz>`.
  Names are matched case-insensitively, and a name may occur only once per line.
  :attr:`coefficient` is a nonzero real number or a fraction :attr:`p/q` of two real numbers, such as ``1/3``, and defaults to 1.
* The operators ``+`` and ``-`` are fields of their own, separated from the terms by whitespace.
* :attr:`weight` is optional, must be a positive number, and defaults to 1.
  It scales the contribution of the combination to the loss.

A combination needs at least two structures and must be balanced in the number of atoms of each type: for every type, the sum of the coefficients times the numbers of atoms of that type vanishes.
The check allows a relative deviation of :math:`10^{-9}`, so a coefficient such as one third has to be written as the fraction ``1/3``.
The target of a combination is the same combination of the target total energies, which are given by the :attr:`energy` fields of the structures.

A field beginning with ``#`` starts a comment that extends to the end of the line.

Example
-------

::

  # two charge states of one structure, which only a qNEP model can tell apart
  defect-chg0 - defect-chg-1
  # an oxygen vacancy against the perfect cell and half an O2 molecule, with weight 2
  vacancy + 0.5*O2 - bulk 2.0
  # a vacancy in a 31-atom cell against a 32-atom perfect cell of one element
  vac31 - 0.96875*bulk32

Training and test combinations
------------------------------

A combination with all its structures in :attr:`train.xyz` is a training combination, which enters the loss function and the column :attr:`rmse_ediff_train` of :ref:`loss.out <loss_out>`.
A combination with all its structures in :attr:`test.xyz` is a test combination, which enters the column :attr:`rmse_ediff_test`.
A combination whose names occur in both files is both a training and a test combination.
A combination with a name that occurs in neither file, or with its structures in different files, is skipped.
``nep`` prints the number of combinations of each kind and a warning with the number of skipped combinations.

Caveats
-------

* A line that does not follow the format, an invalid coefficient or weight, and a combination that is not balanced are input errors.
* A name that occurs on more than one structure of a file, or that violates the rules for labels in :ref:`train.xyz and test.xyz <train_test_xyz>`, is an input error.
