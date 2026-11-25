.. _ediff_in:
.. index::
   single: ediff.in (input file)

``ediff.in``
============

This file lists linear combinations of the total energies of structures, such as energy differences and formation energies, that enter the loss function through the keyword :ref:`lambda_d <kw_lambda_d>`.
It is read only when the keyword :ref:`lambda_d <kw_lambda_d>` is set.

File format
-----------

Each line holds one combination::

  [+|-] <term> {+|- <term>} [w=<weight>]

* A ``term`` is either ``<name>`` or ``<coefficient>*<name>``, without spaces.
  ``name`` is the label given by the ``name=<label>`` field on the comment line of a structure in :ref:`train.xyz and test.xyz <train_test_xyz>`.
  Names are matched case-insensitively for the letters A to Z, and a name may occur only once per line.
  ``coefficient`` is a nonzero real number or a fraction ``p/q`` of two real numbers, such as ``1/3``, and defaults to 1.
* The operators ``+`` and ``-`` are fields of their own, separated from the terms by whitespace.
* ``w=<weight>`` is optional and comes last.
  ``weight`` must be a positive number and defaults to 1.
  It scales the contribution of the combination to the loss.

A combination needs at least two structures and must be balanced in the number of atoms of each type: for every type, the sum of the coefficients times the numbers of atoms of that type vanishes.
The check allows :math:`10^{-6}` atoms of each type to be left over, so a coefficient such as one third is best written as the fraction ``1/3``.
No two structures of a combination may be the same structure for the model, that is have the same types, the same positions to :math:`10^{-5}` Å, the same boundaries, for periodic boundaries the same cell, and for ``model nep_temperature`` the same ``temperature``.
Two such structures can differ at most in ``charge``, which a NEP model ignores and for which a qNEP model predicts no meaningful energy difference.
Atoms are compared in order, so a copy with reordered atoms or with an atom shifted by a lattice vector is not detected.
The target of a combination is the same combination of the target total energies, which are given by the ``energy`` fields of the structures.

A field beginning with ``#`` starts a comment that extends to the end of the line.
A line that breaks any of these rules is an input error.

Example
-------

::

  # oxygen vacancy formation energy against the perfect cell and the O2 molecule
  vac_O + 1/2*O2 - ideal
  # migration barrier of the oxygen vacancy, from the saddle point against the minimum, with weight 2
  vac_O_saddle - vac_O w=2
  # vacancy formation energy in a 31-atom cell against the 32-atom perfect cell of one element
  vac31 - 31/32*bulk32

Training and test combinations
------------------------------

A combination with all its structures in ``train.xyz`` is a training combination, which enters the loss function and the column ``rmse_ediff_train`` of :ref:`loss.out <loss_out>`.
A combination with all its structures in ``test.xyz`` is a test combination, which enters the column ``rmse_ediff_test``.
A combination whose names occur in both files is both a training and a test combination.
A combination with a name that occurs in neither file, or with its structures in different files, is skipped.
``nep`` prints the number of combinations of each kind and a warning with the number of skipped combinations.
