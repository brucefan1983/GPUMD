" Vim syntax file
" Language:          GPUMD Simulation Script File
" Latest Revision:   2025-4-17

syn clear

"-------------------------
" GPUMD core syntax definition
"-------------------------
syn keyword GPUMDpot potential
syn keyword GPUMDsetup replicate velocity correct_velocity potential dftd3 change_box deform
syn keyword GPUMDsetup ensemble fix time_step plumed mc electron_stop add_force add_efield active
syn keyword GPUMDdump dump_netcdf dump_position dump_restart dump_thermo dump_velocity dump_shock_nemd
syn keyword GPUMDdump dump_exyz dump_beads dump_observer dump_dipole dump_polarizability dump_force
syn keyword GPUMDcompute compute compute_cohesive compute_dos compute_elastic
syn keyword GPUMDcompute compute_gkma compute_hac compute_hnema compute_hnemd compute_hnemdec
syn keyword GPUMDcompute compute_phonon compute_sdc compute_msd compute_shc compute_viscosity compute_lsqt compute_rdf
syn keyword GPUMDrun minimize run

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
