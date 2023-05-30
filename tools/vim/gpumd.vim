" Vim syntax file
" Language:          GPUMD Simulation Script File
" Latest Revision:   2023-5-30

syn clear

syn keyword GPUMDpot potential
syn keyword GPUMDsetup change_box ensemble velocity correct_velocity time_step neighbor fix deform
syn keyword GPUMDdump dump_thermo dump_position dump_netcdf dump_restart dump_velocity dump_force active dump_exyz dump_beads dump_observer
syn keyword GPUMDcompute plumed compute compute_shc compute_dos compute_sdc compute_hac compute_hnemd compute_gkma compute_hnema compute_phonon compute_cohesive compute_elastic compute_hnemdec compute_msd compute_viscosity
syn keyword GPUMDrun minimize run
syn keyword GPUMDspecial temperature potential force virial jp jk group num_dos_points f_bin_size bin_size precision observe
syn keyword GPUMDensemble nve nvt_ber nvt_nhc nvt_bdp nvt_lan nvt_bao npt_ber npt_scr heat_nhc heat_bdp heat_lan pimd rpmd trpmd sd

syn keyword NEPsets zbl n_max basis_size l_max neuron force_delta population
syn keyword NEPkws cutoff batch
syn keyword NEPinput version model_type type type_weight
syn keyword NEPrun generation prediction
syn keyword NEPlambda lambda_1 lambda_2 lambda_e lambda_f lambda_v lambda_shear

syn region GPUMDString start=+'+ end=+'+ oneline
syn region GPUMDString start=+"+ end=+"+ oneline

syn match GPUMDNumber "\<[0-9]\+[ij]\=\>"
syn match GPUMDFloat "\<[0-9]\+\.[0-9]*\([edED][-+]\=[0-9]\+\)\=[ij]\=\>"
syn match GPUMDFloat "\.[0-9]\+\([edED][-+]\=[0-9]\+\)\=[ij]\=\>"
syn match GPUMDFloat "\<[0-9]\+[edED][-+]\=[0-9]\+[ij]\=\>"

syn match GPUMDComment "#\(.*&\s*\n\)*.*$"

syn region NEPString start=+'+ end=+'+ oneline
syn region NEPString start=+"+ end=+"+ oneline

syn match NEPNumber "\<[0-9]\+[ij]\=\>"
syn match NEPFloat "\<[0-9]\+\.[0-9]*\([edED][-+]\=[0-9]\+\)\=[ij]\=\>"
syn match NEPFloat "\.[0-9]\+\([edED][-+]\=[0-9]\+\)\=[ij]\=\>"
syn match NEPFloat "\<[0-9]\+[edED][-+]\=[0-9]\+[ij]\=\>"

syn match NEPComment "#\(.*&\s*\n\)*.*$"

if !exists("did_GPUMD_syntax_inits")
  let did_GPUMD_syntax_inits = 1
  hi link GPUMDdump     Function
  hi link GPUMDpot      Typedef
  hi link GPUMDsetup    Typedef
  hi link GPUMDcompute  Define
  hi link GPUMDrun      Statement
  hi link GPUMDspecial  special
  hi link GPUMDensemble ensemble
  hi link GPUMDString   String
  hi link GPUMDNumber   Number
  hi link GPUMDFloat    Float
  hi link NEPrun        Statement
  hi link NEPlambda     special
  hi link NEPrun        Define
  hi link NEPkws        Typedef
  hi link NEPsets       Function
  hi link NEPString     String
  hi link NEPNumber     Number
  hi link NEPFloat      Float

endif

let b:current_syntax = "GPUMD"
