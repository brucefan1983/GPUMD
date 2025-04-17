" Vim syntax file
" Language:          GPUMD Simulation Script File
" Latest Revision:   2025-4-17

syn clear

"-------------------------
" GPUMD core syntax definition
"-------------------------
syn keyword GPUMDpot potential
syn keyword GPUMDsetup replicate velocity correct_velocity potential dftd3 change_box deform ensemble fix time_step plumed mc electron_stop add_force add_efield active
syn keyword GPUMDdump dump_exyz dump_beads dump_observer dump_dipole dump_polarizability dump_force dump_netcdf dump_position dump_restart dump_thermo dump_velocity dump_shock_nemd
