GPUMD -- Graphics Processing Units Molecular Dynamics
*****************************************************

This manual describes how to run molecular dynamics simulations and train neuroevolution potential (:term:`NEP`) models with :program:`GPUMD`.
Start with the :doc:`introduction` and :doc:`installation`, then follow the guide for your task:

* :ref:`Run a molecular dynamics simulation <gpumd_executable>` using an existing potential.
* :ref:`Train or evaluate a NEP model <nep_executable>` using reference data.

The :doc:`theory/index` and :ref:`interatomic potential descriptions <potentials>` explain the methods used in these calculations.

.. toctree::
   :maxdepth: 2
   :caption: Main

   introduction
   installation
   theory/index
   potentials/index

.. toctree::
   :maxdepth: 2
   :caption: Interface

   gpumd/index
   nep/index

.. toctree::
   :maxdepth: 2
   :caption: Backmatter

   credits
   bibliography
   publications
   glossary
   genindex
