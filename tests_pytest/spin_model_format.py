"""Independent legacy-to-block fixture conversion for Spin3 compatibility tests."""


def block_model(legacy, *, legacy_name=False):
    lines = legacy.splitlines()
    header = lines[0].split()
    count = int(lines[1].split()[2])
    fields = dict((line.split()[0], line.split()[1:]) for line in lines[2:2 + count])
    cursor = 2 + count
    while not lines[cursor].startswith('ANN '):
        tokens = lines[cursor].split()
        fields[tokens[0]] = tokens[1:]
        cursor += 1
    fields['ann'] = lines[cursor].split()[1:]
    fields['energy_reference'] = fields.pop('spin_baseline')
    fields['spin_mode' if legacy_name else 'spin_version'] = ['3' if legacy_name else '1']
    C = int(fields['spin_compress'][0])
    T = int(header[1])
    nr, na = map(int, fields['n_max'])
    br, ba = map(int, fields['basis_size'])
    descriptor_count = T * T * ((nr + 1) * (br + 1) + (na + 1) * (ba + 1) + C * 9) + 4 * C * C
    body = lines[cursor + 1:]
    neurons = int(fields['ann'][0])
    dim, remainder = divmod(len(body) - descriptor_count - 2 * neurons * T - 1, neurons * T + 1)
    assert remainder == 0
    ann_count = (dim + 2) * neurons * T + 1
    values = body[:-dim]
    offset = ann_count
    for basis_count in ((nr + 1) * (br + 1), (na + 1) * (ba + 1)):
        old = values[offset:offset + T * T * basis_count]
        values[offset:offset + len(old)] = [old[b * T * T + pair]
            for pair in range(T * T) for b in range(basis_count)]
        offset += len(old)
    result = ['nep ' + ' '.join(header[1:]), 'format_version 1',
              'model nep_spin3' if legacy_name else 'model nep_spin']
    result += [key + ' ' + ' '.join(value) for key, value in fields.items()]
    result += [f'@ parameters {len(values)}', *values, f'@ q_scaler {dim}', *body[-dim:]]
    return '\n'.join(result) + '\n'
