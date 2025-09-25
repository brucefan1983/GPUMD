.. _kw_fine_tune:
.. index::
   single: fine_tune (keyword in nep.in)

:attr:`fine_tune`
=================

This keyword instructs :program:`nep` to fine tune a model starting from a foundation model.
The syntax is::

  fine_tune <nep_model_file> <nep_restart_file>

where :attr:`<nep_model_file>` is the potential file for the foundation model and :attr:`<nep_restart_file>` is the restart file for the foundation model.

Currently, we provide one foundation model in GPUMD/potentials/nep/nep89_20250409.

For more details, see the following exemplary :attr:`nep.in` file::

  # for prediction
  #type       89 H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Ac Th Pa U Np Pu 
  #prediction 1

  # for fine-tuning
  fine_tune nep89_20250409.txt nep89_20250409.restart
  type <your types>

  # These cannot be changed:
  version    4
  zbl        2
  cutoff     6 5
  n_max      4 4
  basis_size 8 8
  l_max      4 2 1
  neuron     80

  # These can be changed:
  lambda_1   0
  lambda_e   1
  lambda_f   1
  lambda_v   1
  batch      5000
  population 50
  save_potential 1000 0
  generation 5000
  
