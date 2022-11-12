.. _virial_out:
.. index::
   single: virial_train.out (output file)
   single: virial_test.out (output file)

``virial_*.out``
================

The ``virial_train.out`` and ``virial_test.out`` files contain the predicted and target virials.
There are 2 columns.
The first column gives the virials in units of eV/atom calculated using the :term:`NEP` model.
The second column gives the corresponding target virials in units of eV/atom.

The are :math:`6N_\mathrm{c}` rows, where :math:`N_\mathrm{c}` is the number of configurations in the :ref:`train.xyz and test.xyz input files <train_test_xyz>`.

* The first :math:`N_\mathrm{c}` rows correspond to the :math:`xx` component of the virial.
* The second :math:`N_\mathrm{c}` rows correspond to the :math:`yy` component of the virial.
* The third :math:`N_\mathrm{c}` rows correspond to the :math:`zz` component of the virial.
* The fourth :math:`N_\mathrm{c}` rows correspond to the :math:`xy` component of the virial.
* The fifth :math:`N_\mathrm{c}` rows correspond to the :math:`yz` component of the virial.
* The sixth :math:`N_\mathrm{c}` rows correspond to the :math:`zx` component of the virial.
