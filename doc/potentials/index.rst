.. _potentials:
.. index::
   single: Interaction models
   single: Interatomic potentials

Interatomic potentials
**********************

This chapter describes the supported potential forms and their parameter files.
Load a model with the :ref:`potential command <kw_potential>`; the file identifies the model and its elements.
Use the parameter order given on the corresponding model page.

* **Machine-learned models:** :ref:`NEP <nep_formalism>` and :ref:`NEP-spin <nep_spin>`.
  Training is described in :ref:`the NEP chapter <nep_executable>`.
* **Analytical and tabulated models:** Tersoff, Lennard-Jones, EAM, ADP, and force constant potentials.
* **Layered materials:** NEP+ILP, SW+ILP, and Tersoff+ILP combine separate intralayer and interlayer interactions.
  Their layer assignments and auxiliary files are part of the model setup.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   tersoff_1988
   tersoff_1989
   tersoff_mini
   lennard_jones
   eam
   fcp
   nep
   nep_spin
   nep_ilp
   sw_ilp
   tersoff_ilp
   adp
