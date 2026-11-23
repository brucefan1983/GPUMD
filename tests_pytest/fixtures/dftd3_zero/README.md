# Fixtures of test_dftd3_zero.py (DFT-D3 zero damping, `dftd3 <f> <rc> <rc_cn> [zero|bj]`)

| File | Content | Produced by |
|---|---|---|
| `all_functionals.json`, `hco_mix.xyz` | simple-dftd3 D3(0) energy/forces of all 48 table functionals on one H/C/O cell | `generate_all_functionals.py` |
| `zero_baseline.json` | median/max errors of this implementation per `reference.json` case (tolerance baselines) | `measure_zero_baseline.py` |
| `reference.json` | simple-dftd3 D3(0) energy/forces/virial (two-body) for PBE (8 structures), revPBE, BLYP, B3LYP, TPSS (3 structures each) | `generate_reference.py` |
| `heavy_cluster.xyz` | periodic 10 A cell with La, Hg, Tl, Bi, Pt, Os, W, Cs, Xe, I (Z <= 83; probes more R0 table entries) | `generate_reference.py` |
| `bj_frozen.json`, `tiny24.xyz` | output of the unmodified upstream binary (84fd855e, BJ damping only) for `dftd3 pbe rc rcn` (frozen reference) | `generate_bj_frozen.py` |
| `bj_spread.json` | run-to-run spread of that unchanged binary per structure (60 repeats + the frozen run) | `generate_bj_frozen.py spread` |
| `zero_nep_H_C_O_Cu.txt` | all-zero NEP (H C O Cu): output of a run = D3 contribution alone, bit-reproducible | `make_zero_nep.py` |
| `*.xyz` (others) | structures of the reference cases (`benzene_dimer`, `graphite`, `water27`, `water27_rc25`, `water27_tri`, `water_large`, `cu_large`) | stored as is |

## Reference values (`reference.json`)
- Oracle: simple-dftd3 (python `dftd3`, version recorded in `reference.json`), `ZeroDampingParam` with
  explicit parameters s6 = 1, sr8 = 1, alpha6 = 14 (alpha8 = 16) and (sr6, s8) of the functional,
  `s9 = 0` (no ATM). The script checks that the explicit parameters equal simple-dftd3's own
  parameters of the same functional (simple-dftd3 names `b-lyp`, `b3-lyp` = GPUMD names `blyp`, `b3lyp`).
- Parameters (def2-QZVP set, Grimme et al., J. Chem. Phys. 132, 154104 (2010)): PBE sr6 = 1.217, s8 = 0.722;
  revPBE 0.923 / 1.010; BLYP 1.094 / 1.682; B3LYP 1.261 / 1.703; TPSS 1.166 / 1.105.
- Real-space cutoffs equal to those given to GPUMD (pair = `rc_potential`, CN = `rc_cn`), as in the BJ test.
- **Length constant.** The code's `Bohr` macro (`src/force/dftd3/dftd3para.cuh`) is the float constant
  0.5291772575069165f, used for all lengths (positions, cutoffs, gradient and virial conversion) in the
  reference. simple-dftd3 uses CODATA2018 (0.529177210903 A) internally, including its pair radii R0. The oracle
  therefore rescales `sr6` and `sr8` by the ratio 0.529177210903/0.5291772575069165, so that r/R0 equals the ratio
  of Angstrom quantities (the R0 table of GPUMD is in Angstrom). Without this the energy differs by up to 1.5e-6
  relative. Residual sub-1e-6 differences from the covalent-radius table of simple-dftd3 (CN) are not removed by
  the rescaling.
- Units: eV, eV/A, virial = -sigma (GPUMD convention), Hartree = 27.211386245988 eV.

## Tolerances
`min(cap, max(2 x BASELINE, floor))`. Caps: relative 2e-6 of |E_D3| and of the largest virial component,
1e-5 eV/A for forces. Floors: 2e-6 eV, 1e-6 eV/A, 2e-5 eV (virial). The baselines (`ZERO_BASELINE`) are the median errors of THIS implementation against `reference.json`
(energy, largest force component, largest virial component, through the same code path as the test,
`d3_part`), stored in `zero_baseline.json` and written by `measure_zero_baseline.py` (5 repeats; binary md5,
GPU, CUDA version and date are recorded in the file). Consequently the tests detect changes of the
implementation relative to its measured behaviour and bound the error by the caps; they are not an
independent proof of accuracy beyond the caps. The measured median errors are 0.12 to 0.84 of the cap
(largest: graphite virial, 0.84, where 2 x baseline exceeds the cap so the cap applies); the maximum over
repeats equals the median (no run-to-run spread seen). Where 2 x baseline is below the floor, the floor is
used.

## All zero-damping functionals (`all_functionals.json`, `hco_mix.xyz`)
- Produced by `generate_all_functionals.py` (simple-dftd3 1.7.0, dftd3-python, env with numpy and `dftd3`). Structure
  `hco_mix.xyz`: H2O + H2CO + CH4 in a periodic 11 A cube; cutoffs 12 / 6 A; the test uses the all-zero NEP.
  Energy and forces only (tolerance min(2e-6 |E|, 2e-6 eV) and 1e-6 eV/A; no baseline entry). Measured margin (3 repeats): energy error is at most 1/2.7 of the tolerance (worst: pbe0), force error at most 1/2.0 (worst: slaterdiracexchange).
- One entry per functional of `src/force/dftd3/dftd3zero_table.cuh` (48). The parameters (s6, sr6, s8, sr8; alpha6 = 14)
  are parsed from that table and passed explicitly (same length-constant treatment as above).
- Name mapping GPUMD -> simple-dftd3 (derived by the script, stored in `_name_map`): the GPUMD name is tried first, then
  the names with one or two '-' inserted; a candidate is accepted only if simple-dftd3's own parameters reproduce the
  energy of the explicit table parameters (relative 1e-10). Functionals without an accepted candidate are listed in
  `_unmapped` and have no reference entry. Result: all 48 are accepted, each under its GPUMD name (simple-dftd3 ignores
  '-' in names), none unmapped.

## BJ unchanged (`bj_frozen.json`, `bj_spread.json`)
- The reference is the output (dump lines with `precision double`: energy, virial, species, positions, forces) of a
  `dftd3 pbe 12 6` run with the all-zero NEP, so it is the D3 contribution alone. The `stress` field of the header is
  removed (varies from run to run on the same binary).
- Comparison. tiny24 and benzene_dimer: the unchanged baseline gave one single output over 60 repeats (also 40 in an
  earlier measurement), so the test is bitwise. water27, graphite, water27_tri: the UNCHANGED baseline fluctuates
  in the last digits from run to run, because energy, virial and forces are summed with double `atomicAdd` in a
  thread order that is not fixed. There the test requires |new - frozen| <= 3 x spread, per group (energy, largest
  virial component, largest force component), spread = max - min over the 60 repeats of the baseline binary and the
  frozen run (`bj_spread.json`; measured values 1e-19 to 3e-17, i.e. last-digit level). The factor 3 is a safety
  factor for the finite number of repeats; a group with spread 0 must agree exactly. The test default-vs-explicit
  `bj` uses the same criterion. No tolerance is chosen by hand.
- Binary: built from upstream/master 84fd855e with `src/` unmodified (sha256 of `dftd3.cu` is in the JSON),
  `make CUDA_ARCH="-arch=sm_86"`, CUDA 12.3, RTX A6000.
- The reference is hardware dependent in principle (the same instruction sequence on other GPUs may round
  differently); if the test fails on a different GPU/CUDA version only, regenerate by running the script with the
  unchanged baseline binary on that machine (`spread` as well), never with the new binary.
