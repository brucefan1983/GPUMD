.. index::
   single: gpumd input files

Input files
===========

Run ``gpumd`` from a directory containing:

* :ref:`model.xyz <model_xyz>`: the initial atomic configuration and simulation cell.
* :ref:`run.in <run_in>`: the simulation commands, in execution order.

The :ref:`potential command <kw_potential>` in ``run.in`` specifies the potential files to load.
Relative file paths are resolved from this working directory.
The other input files below are needed only for the corresponding calculations.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   model_xyz
   run_in
   replica_temperatures
   kpoints_in
   eigenvector_in
