" Vim syntax file
" Language:          GPUMD Simulation Script File
" Maintainer:        Ke Xu <twtdq@stu.xmu.edu.cn>
" Latest Revision:   2021-10-12

syn clear

syn keyword GPUMDpot potential
syn keyword GPUMDsetup ensemble velocity time_step neighbor fix deform cutoff delta
syn keyword GPUMDdump dump_thermo dump_position dump_netcdf dump_restart dump_velocity dump_force
syn keyword GPUMDcompute compute compute_shc compute_dos compute_sdc compute_hac compute_hnemd compute_gkma compute_hnema
syn keyword GPUMDrun minimize run

syn keyword GPUMDspecial temperature potential force virial jp jk group num_dos_points f_bin_size bin_size

syn keyword GPUMDensemble nve nvt_ber nvt_nhc nvt_bdp nvt_lan npt_ber heat_nhc heat_bdp heat_lan sd

syn region GPUMDString start=+'+ end=+'+ oneline
syn region GPUMDString start=+"+ end=+"+ oneline

syn match GPUMDNumber "\<[0-9]\+[ij]\=\>"
syn match GPUMDFloat "\<[0-9]\+\.[0-9]*\([edED][-+]\=[0-9]\+\)\=[ij]\=\>"
syn match GPUMDFloat "\.[0-9]\+\([edED][-+]\=[0-9]\+\)\=[ij]\=\>"
syn match GPUMDFloat "\<[0-9]\+[edED][-+]\=[0-9]\+[ij]\=\>"

syn match GPUMDComment "#\(.*&\s*\n\)*.*$"

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
endif

let b:current_syntax = "GPUMD"
