# qNEP training

This example trains a charge-aware PbTe potential using `charge_mode 2`.

Run:

```bash
/path/to/nep
python plot_results.py
```

The main outputs are `nep.txt`, `nep.restart`, `loss.out`, and the training-set prediction files. `plot_results.py` writes `energy_parity.png`, `force_parity.png`, and `training_history.png`.
