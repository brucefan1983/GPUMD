"""Analytic zero or E_i = tanh(0.1 |S_i|^2) potential in the current Spin format."""
from spin_fixtures import FIXTURE

def analytic_model(onsite):
    lines = (FIXTURE / "nep_spin_o3c2_uniform.nep").read_text().splitlines()
    parameter = next(i for i, line in enumerate(lines) if line.startswith("@ parameters "))
    scaler = next(i for i, line in enumerate(lines) if line.startswith("@ q_scaler "))
    count = int(lines[parameter].split()[2])
    dim = int(lines[scaler].split()[2])
    values = [0.0] * count
    if onsite:
        for element in range(2):
            offset = element * 30 * (dim + 2)
            values[offset + 30] = 0.1
            values[offset + 30 * dim + 30] = 1.0
    header = ["energy_reference 0 0" if line.startswith("energy_reference ")
              else "spin_type Fe Ge" if line.startswith("spin_type ") else line
              for line in lines[:parameter]]
    return "\n".join(header + [f"@ parameters {count}"] + list(map(str, values))
                     + [f"@ q_scaler {dim}"] + ["1.0"] * dim) + "\n"

def structure(spins=None):
    lines = (FIXTURE / 'model_large_box.xyz').read_text().splitlines()
    lines[1] = lines[1].replace(':spin:R:3', ':vel:R:3:spin:R:3:spin_vel:R:3')
    for i in range(4):
        f = lines[i+2].split()
        spin = spins[i] if spins is not None else f[4:7]
        lines[i+2] = ' '.join(f[:4] + ['0.001', '-0.002', '0.003'] + list(map(str, spin)) + ['0.02','-0.03','0.04'])
    return '\n'.join(lines)+'\n'
