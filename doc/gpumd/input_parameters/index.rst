.. _gpumd_input_parameters:
.. index::
   single: gpumd input parameters

Input parameters
================

The keywords below are grouped by purpose.
See :ref:`run.in <run_in>` for command order and how settings apply to successive runs.

.. rubric:: Simulation setup and execution

.. toctree::
   :maxdepth: 1

   replicate
   potential
   dftd3
   kspace
   velocity
   time_step
   run

.. rubric:: Minimization and static calculations

.. toctree::
   :maxdepth: 1

   minimize
   minimize_spin_fire
   change_box
   compute_cohesive
   compute_elastic
   compute_hessian
   compute_phonon

.. rubric:: Ensembles

.. toctree::
   :maxdepth: 1

   ensemble
   ensemble_standard
   ensemble_mttk
   ensemble_heat
   ensemble_ttm
   ensemble_pimd
   ensemble_qtb
   ensemble_shock_nemd
   ensemble_msst
   ensemble_nphug
   ensemble_sib
   ensemble_tspin
   ensemble_glsd

.. rubric:: Thermodynamic integration

.. toctree::
   :maxdepth: 1

   ensemble_ti
   ensemble_ti_liquid
   ensemble_ti_spring
   ensemble_ti_as
   ensemble_ti_rs
   ensemble_ti_nep

.. rubric:: Forces, constraints, and deformation

.. toctree::
   :maxdepth: 1

   add_force
   add_efield
   add_mfield
   add_spring
   fix
   move
   correct_velocity
   deform
   deposit
   electron_stop

.. rubric:: Monte Carlo, biased sampling, and replica dynamics

.. toctree::
   :maxdepth: 1

   mc
   plumed
   multi_replica

.. rubric:: Thermodynamic and structural properties

.. toctree::
   :maxdepth: 1

   compute
   compute_chunk
   compute_rdf
   compute_angular_rdf
   compute_adf
   compute_orientorder

.. rubric:: Transport and time correlations

.. toctree::
   :maxdepth: 1

   compute_sdc
   compute_msd
   compute_ic
   compute_dos
   compute_hac
   compute_hnemd
   compute_hnemdec
   compute_shc
   compute_gkma
   compute_hnema
   compute_viscosity
   compute_lsqt
   compute_dpdt

.. rubric:: Model evaluation and active learning

.. toctree::
   :maxdepth: 1

   active
   compute_extrapolation
   dump_observer

.. rubric:: Output

.. toctree::
   :maxdepth: 1

   dump_thermo
   dump_xyz
   dump_beads
   dump_netcdf
   dump_restart
   dump_dipole
   dump_polarizability
   dump_shock_nemd
