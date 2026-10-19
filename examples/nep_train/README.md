# NEP training

This example trains a standard NEP potential for PbTe from `train.xyz` for 20,000 generations.

Run:

```bash
/path/to/nep
python3 plot_results.py
```

The main outputs are `nep.txt`, `nep.restart`, `loss.out`, and the training-set prediction files. `plot_results.py` writes `energy_parity.png`, `force_parity.png`, and `training_history.png`.
