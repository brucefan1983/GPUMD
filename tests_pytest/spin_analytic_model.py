"""Small analytic potential for independent Spin runtime tests."""
from spin_fixtures import FIXTURE

def analytic_model(onsite):
    """E_i = tanh(0.1 |S_i|^2), or zero, using the actual Spin3 ANN path."""
    lines = (FIXTURE / 'nep4_spin3_o3c2_uniform.nep').read_text().splitlines()
    start = next(i+1 for i, s in enumerate(lines) if s.startswith('ANN '))
    # This fixture has 2 types, 30 neurons, 360 structural, 72 spin and 16 projection weights.
    dim, remainder = divmod(len(lines)-start-569, 61)
    assert remainder == 0 and dim > 30
    values = [0.0] * (len(lines)-start)
    values[-dim:] = [1.0] * dim
    if onsite:
        for t in range(2):
            offset = t * 30 * (dim+2)
            values[offset+30] = 0.1  # 5 radial + 5 * (4 + q222) structural channels
            values[offset+30*dim+30] = 1.0
    lines[2] = 'spin_baseline 0 0'
    return '\n'.join(lines[:start] + [str(x) for x in values]) + '\n'

def structure(spins=None):
    lines = (FIXTURE / 'model_large_box.xyz').read_text().splitlines()
    lines[1] = lines[1].replace(':spin:R:3', ':vel:R:3:spin:R:3:spin_vel:R:3')
    for i in range(4):
        f = lines[i+2].split()
        spin = spins[i] if spins is not None else f[4:7]
        lines[i+2] = ' '.join(f[:4] + ['0.001', '-0.002', '0.003'] + list(map(str, spin)) + ['0.02','-0.03','0.04'])
    return '\n'.join(lines)+'\n'
