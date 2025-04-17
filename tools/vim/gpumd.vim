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
syn keyword GPUMDcompute compute compute_cohesive compute_dos compute_elastic compute_gkma compute_hac compute_hnema compute_hnemd compute_hnemdec compute_phonon compute_sdc compute_msd compute_shc compute_viscosity compute_lsqt compute_rdf
syn keyword GPUMDrun minimize run
" syn keyword GPUMDspecial seed dp pbe temp tperiod pperiod x y z aniso tswitch tequil lambda vp k thickness qmass mu group kappa temperature potential force virial jp jk num_dos_points cubic bin_size f_bin_size atom observe average precision single double interval bin_size
" syn keyword GPUMDensemble nve nvt_ber nvt_nhc nvt_bdp nvt_lan nvt_bao npt_ber npt_scr npt_mttk heat_nhc heat_bdp heat_lan pimd rpmd trpmd ti_spring ti_as ti_rs ti wall_piston wall_mirror wall_harmonic msst nphug canonical sgc vcsgc sd fire

"-------------------------
" NEP related syntax definition
"-------------------------
syn keyword NEPsets zbl use_typewise_cutoff_zbl use_typewise_cutoff n_max basis_size l_max neuron force_delta population
syn keyword NEPkws cutoff batch
syn keyword NEPinput version model_type type type_weight
syn keyword NEPrun generation prediction
syn keyword NEPlambda lambda_1 lambda_2 lambda_e lambda_f lambda_v lambda_shear

"-------------------------
" GPUMD syntax rules (strings/numbers/comments)
"-------------------------
syn region GPUMDString start=+'+ end=+'+ oneline
syn region GPUMDString start=+"+ end=+"+ oneline
syn match GPUMDNumber "\<[0-9]\+[ij]\=\>"
syn match GPUMDFloat "\<[0-9]\+\.[0-9]*\([edED][-+]\=[0-9]\+\)\=[ij]\=\>"
syn match GPUMDFloat "\.[0-9]\+\([edED][-+]\=[0-9]\+\)\=[ij]\=\>"
syn match GPUMDFloat "\<[0-9]\+[edED][-+]\=[0-9]\+[ij]\=\>"
syn match GPUMDComment "#\(.*&\s*\n\)*.*$"

"-------------------------
" NEP syntax rules (strings/numbers/comments)
"-------------------------
syn region NEPString start=+'+ end=+'+ oneline
syn region NEPString start=+"+ end=+"+ oneline
syn match NEPNumber "\<[0-9]\+[ij]\=\>"
syn match NEPFloat "\<[0-9]\+\.[0-9]*\([edED][-+]\=[0-9]\+\)\=[ij]\=\>"
syn match NEPFloat "\.[0-9]\+\([edED][-+]\=[0-9]\+\)\=[ij]\=\>"
syn match NEPFloat "\<[0-9]\+[edED][-+]\=[0-9]\+[ij]\=\>"
syn match NEPComment "#\(.*&\s*\n\)*.*$"

"-------------------------
" Highlight link definition
"-------------------------
if !exists("did_GPUMD_syntax_inits")
  let did_GPUMD_syntax_inits = 1
  hi link GPUMDdump     Function
  hi link GPUMDpot      Typedef
  hi link GPUMDsetup    Typedef
  hi link GPUMDcompute  Define
  hi link GPUMDrun      Statement
" hi link GPUMDspecial  Special
" hi link GPUMDensemble Constant
  hi link GPUMDString   String
  hi link GPUMDNumber   Number
  hi link GPUMDFloat    Float
  hi link GPUMDComment  Comment

  hi link NEPsets       Function
  hi link NEPkws        Typedef
  hi link NEPinput      Define
  hi link NEPrun        Statement
  hi link NEPlambda     Special
  hi link NEPString     String
  hi link NEPNumber     Number
  hi link NEPFloat      Float
  hi link NEPComment    Comment
endif

let b:current_syntax = "GPUMD"
