# NEP training

This example trains a standard NEP potential for PbTe from `train.xyz` for 10,000 generations followed by another 10,000 generations of continued training.

Run:

```bash
/path/to/nep
python3 plot_results.py
```

The main outputs are `nep.txt`, `nep.restart`, `loss.out`, and the training-set prediction files. `plot_results.py` writes `energy_parity.png`, `force_parity.png`, and `training_history.png`.
