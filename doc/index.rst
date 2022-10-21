GPUMD -- Graphics Processing Units Molecular Dynamics
*****************************************************

:program:`GPUMD` stands for *Graphics Processing Units Molecular Dynamics*.
It is a general-purpose molecular dynamics (:term:`MD`) package fully implemented on graphics processing units (:term:`GPU`).
It is written in CUDA C++ and requires a CUDA-enabled Nvidia GPU of compute capability no less than 3.5.

It has native support for neuroevolution potential (:term:`NEP`) machine learning potentials, including both their construction and use in :term:`MD` simulations.
It is also highly efficient for conducting :term:`MD` simulations with many-body potentials such as the Tersoff potential and is particularly good for heat transport applications.

There are several packages that provide Python interfaces to :program:`GPUMD` functionality, including `calorine <https://gitlab.com/materials-modeling/calorine>`_, `gpyumd <https://github.com/AlexGabourie/gpyumd>`_, and `pyNEP <https://github.com/bigd4/PyNEP>`_.

.. toctree::
   :maxdepth: 2
   :caption: Main

   installation
   theory/index
   potentials/index
   tutorials/index

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
   glossary
   genindex
