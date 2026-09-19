# Durable behavior-contract diagnostics

`full` is the only acceptance suite. It runs all 175 manifest cases and then
evaluates all 18 cross-case relations for both the baseline and candidate:

```text
Summary: 175 passed, 0 failed; relations: 18 passed, 0 failed, 0 skipped
```

The 20 focused suites are diagnostic views of the same manifest. They may skip
relations when not all members are selected and therefore do not replace the
`full` acceptance run.

## Diagnostic suites

| Suite | Behavior covered |
|---|---|
| `quick` | representative smoke subset |
| `standard` | conventional NVE, NVT, and NPT dynamics |
| `mttk` | MTTK thermostat/barostat and pressure coupling |
| `adaptive_time_step` | displacement-limited step adjustment |
| `heat` | heat baths, fixed-power transport, and TTM workflows |
| `pimd` | PIMD state, RPMD/TRPMD transitions, and bead output |
| `special` | MSST, NPHug, and moving-wall ensembles |
| `ti` | thermodynamic-integration variants |
| `lifecycle` | multiple runs and setting propagation/reset |
| `lifecycle_contract` | ensemble/run ordering requirements |
| `parser` | lexical and top-level command parsing |
| `parsing_validation` | option grammar, duplicate/conflicting options, and positive intervals |
| `invalid` | intentionally rejected inputs and diagnostics |
| `actions` | Action ordering, execution, output, and cleanup |
| `potential` | potential initialization, transition, and consumers |
| `snapshot` | supported commands that depend on later input settings |
| `state` | force/integrator state across runs |
| `first_step_init` | initialization consumed by the first integration step |
| `static` | non-dynamics calculations |
| `transport` | transport measurements |

## Cross-case relation oracles

All relations are evaluated independently for baseline and candidate. They
express stronger behavior than baseline/candidate equality alone because they
compare different inputs that should be physically or operationally
equivalent.

| Relation | Oracle |
|---|---|
| `neighbor_alias_translation_invariance` | translated two-atom periodic systems produce identical thermodynamics |
| `compute_chunk_output_concatenation` | multiple chunk Actions produce the ordered concatenation of standalone outputs |
| `compute_chunk_restart_equality` | standalone and combined chunk Actions leave the same restart state |
| `dos_output_equality` | DOS output is unchanged by adding SDC or reversing Action order |
| `mvac_output_equality` | MVAC output is unchanged by adding SDC or reversing Action order |
| `sdc_output_equality` | SDC output is unchanged by adding DOS or reversing Action order |
| `dos_sdc_thermo_equality` | DOS/SDC combinations leave thermodynamics unchanged |
| `dos_sdc_restart_equality` | DOS/SDC combinations leave restart state unchanged |
| `phonon_d_output_equality` | leading comments/blanks before `replicate` do not change the dynamical matrix |
| `phonon_omega2_output_equality` | leading comments/blanks before `replicate` do not change phonon frequencies |
| `hnemdec_onsager_placement_equality` | HNEMDEC placement before or after its ensemble gives identical Onsager output |
| `hnemdec_thermo_placement_equality` | HNEMDEC placement does not change thermodynamics |
| `hnemdec_restart_placement_equality` | HNEMDEC placement does not change restart state |
| `observer_first_run_output_equality` | implicit and explicit observer controls agree in the first run |
| `observer_first_run_xyz_equality` | implicit and explicit observer controls emit the same first-run trajectory |
| `observer_reset_second_run_equality` | average-observer state resets before the second run |
| `observer_reset_restart_equality` | observer reset leaves the expected restart state |
| `msd_option_order_equality` | supported MSD option orders produce identical output |

Several negative cases complement these relations. In particular, they
preserve sequential validation of late or duplicate `replicate`, duplicate
`dftd3`/`kspace`, malformed optional arguments, conflicting MSD selections,
and zero or negative output intervals. Cases that reject a command after a
completed run also verify that the earlier run finished before the later error
was reported.

To inspect a failure with a focused suite:

```bash
python3 run_regression.py \
  --repo-root /path/to/GPUMD \
  --baseline /path/to/gpumd_baseline \
  --candidate /path/to/gpumd_candidate \
  --suite parsing_validation \
  --keep-all
```

Acceptance always uses `--suite full`.
