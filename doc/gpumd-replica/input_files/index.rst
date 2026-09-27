.. index::
   single: gpumd_replica input files

Input files
===========

A simulation requires ``run.in``, ``model.xyz``, and a :ref:`supported potential file <replica_kw_potential>` in the working directory.
Run ``gpumd_replica`` from this directory.
One process uses the visible GPUs on a single node; select GPUs with ``CUDA_VISIBLE_DEVICES`` or the job scheduler.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   run_in
   model_xyz
   temperatures
