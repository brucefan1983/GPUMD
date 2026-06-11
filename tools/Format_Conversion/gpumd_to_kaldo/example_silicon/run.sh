#!/usr/bin/env bash
set -euo pipefail
python ../gpumd_to_kaldo.py \
  --nep ../../../../potentials/nep/Si_2022_NEP4_4body.txt \
  --supercell 3 3 3 --acoustic-sum --out gpumd_fc.npz
echo "Load in kaldo: ForceConstants.from_folder('.', format='gpumd')"
