#!/usr/bin/env python3
"""Freeze FP64/autograd spin oracles from the independent BFO prototype.

Requires torch, e3nn, ASE and --reference-dir containing prototype.py,
bfo_models.py and its frozen support/reference/torchnep sources. The native
C++ tables and kernels are deliberately never imported by this generator.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_dir))
    import bfo_models as b
    import prototype as p
    from torchnep.model import FittingNet
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {str(f.relative_to(args.reference_dir)): hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in args.reference_dir.rglob('*.py') if '__pycache__' not in str(f)}
    class SpinModel(p.DensityNEP):
        def make_cache(self, r, z):
            return b.structure(self.base, r, z, self.batch, True)
    for C, L, S in [(3, 2, 2), (2, 4, 4), (1, 0, 1)]:
        torch.manual_seed(3401 + C + L + S)
        p.CONFIG.update({k:v for k,v in b.CONFIG.items() if not k.startswith('spin_')})
        p.CONFIG['spin_mode'] = 0
        model = SpinModel(False, C, L, S)
        model.base.fitting_nets = torch.nn.ModuleList([FittingNet(model.dim,32) for _ in b.TYPES])
        with torch.no_grad():
            model.base.c_param_2.normal_(0,.08)
            model.base.c_param_3.normal_(0,.08)
            model.base.b1.fill_(.021)
            # Nonunit scalers test the readout and all chain-rule derivatives.
            model.base.q_scaler.copy_(torch.linspace(.25,1.2,model.dim))
        name = f'c{C}l{L}s{S}'
        base = model.base
        parameters = []
        for net in base.fitting_nets:
            for t in (net.w0.T, net.b0, net.w1): parameters.extend(t.detach().flatten().tolist())
        for t in (base.b1, base.c_param_2, base.c_param_3, model.radial_mix, model.weights, model.center_mix):
            parameters.extend(t.detach().flatten().tolist())
        # Models are stored as FP32: round the oracle's parameters/scalers too.
        with torch.no_grad():
            for t in model.parameters(): t.copy_(t.float().double())
            base.q_scaler.copy_(base.q_scaler.float().double())
        header = f"""nep 3 Fe Bi O
format_version 1
model nep_spin
spin_version 1
cutoff 6 5 128 128
n_max 4 4
basis_size 8 8
l_max 4 2 0
ann 32 0
spin_basis_size 8
spin_l_max {L}
spin_compress {C}
spin_order {S}
spin_soc 1
spin_scaler 1
spin_type Fe
spin_cutoff 6
energy_reference 0 0 0
"""
        text = header + f'@ parameters {len(parameters)}\n' + ''.join(f'{x:.17g}\n' for x in parameters)
        text += f'@ q_scaler {model.dim}\n' + ''.join(f'{x:.17g}\n' for x in base.q_scaler.tolist())
        (args.output/(name+'.nep')).write_text(text)
        cases = {}
        positions = np.array([[.2,.3,.4],[2.1,.5,.7],[.8,2.4,.6],[1.8,1.9,2.2]])
        spins = np.array([[.8,-.4,1.1],[-.5,.9,.3],[.2,.8,-.4],[-.7,.1,.2]])
        for case_name, cell, r0, s0 in [
            ('large', np.diag([24.,25.,26.]),positions,spins),
            ('triclinic', np.array([[4.4,0,0],[.7,4.8,0],[.2,.6,4.5]]),positions,spins),
            ('zero', np.diag([24.,25.,26.]),positions,np.zeros_like(spins)),
            ('isolated', np.diag([40.,40.,40.]),positions+np.arange(4)[:,None]*7,spins),
        ]:
            atoms = Atoms(['Fe','Fe','Bi','O'],positions=r0,cell=cell,pbc=True)
            i,j,sh = neighbor_list('ijS',atoms,6.)
            r = torch.tensor(r0[None],requires_grad=True)
            sp = torch.tensor(s0[None],requires_grad=True)
            strain = torch.zeros(3,3,requires_grad=True)
            deform = torch.eye(3)+strain
            z = torch.tensor([[0,0,1,2]])
            model.batch = {'i':torch.tensor(i), 'j':torch.tensor(j),
                           'shift':torch.tensor(sh@cell)@deform}
            q = model.features(r@deform,sp,z,True).reshape(-1,model.dim)
            energy = p.readout(base,q,z)+0*(r.sum()+sp.sum()+strain.sum())
            gr,gs,gv = torch.autograd.grad(energy.sum(),(r,sp,strain))
            cases[case_name] = dict(types=['Fe','Fe','Bi','O'],positions=r0.tolist(),spins=s0.tolist(),
                cell=cell.flatten().tolist(),energy=energy.sum().item(),potential=energy.detach().tolist(),
                force=(-gr[0]).tolist(),mforce=(-gs[0]).tolist(),virial=(-gv).flatten().tolist())
        result = dict(model_sha256=hashlib.sha256(text.encode()).hexdigest(),dimension=model.dim,
                      parameters=len(parameters),cases=cases,source_sha256=provenance)
        (args.output/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(name,model.dim,len(parameters))

if __name__ == '__main__': main()
