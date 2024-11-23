Introduction
============

:program:`GPUMD` stands for *Graphics Processing Units Molecular Dynamics*.
It is a general-purpose molecular dynamics (:term:`MD`) package fully implemented on graphics processing units (:term:`GPU`).
In addition to :ref:`several empirical interatomic potentials <potentials>`, it supports :ref:`neuroevolution potential <nep_formalism>` (:term:`NEP`) models.
:program:`GPUMD` also allows one to construct the latter type of models using the :ref:`nep executable <nep_executable>`.

:program:`GPUMD` is also highly efficient for conducting :term:`MD` simulations with many-body potentials such as the Tersoff potential and is particularly good for heat transport applications.
It is written in CUDA C++ and requires a CUDA-enabled Nvidia GPU of compute capability no less than 3.5.

Python interfaces
-----------------
There are several packages that provide Python interfaces to :program:`GPUMD` functionality, including `calorine <https://calorine.materialsmodeling.org>`_, `gpyumd <https://github.com/AlexGabourie/gpyumd>`_, and `pyNEP <https://github.com/bigd4/PyNEP>`_.

Github Discussions
------------------
* For general discussions: https://github.com/brucefan1983/GPUMD/discussions
* For issue reporting: https://github.com/brucefan1983/GPUMD/issues

Discussion groups
-----------------
There is a Chinese discussion group based on the following QQ number: 778083814
