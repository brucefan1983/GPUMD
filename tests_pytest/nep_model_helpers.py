"""Model-file helpers shared by the NEP runtime tests."""


def write_scaled_nep_model(model_path, path, factor=1.05):
    """Scale parameters and scalers in a one-value-per-line NEP test fixture.

    Scaling both blocks preserves the behavior of the original legacy-file tests.
    Metadata and energy references are kept unchanged.
    """
    lines = model_path.read_text().splitlines()
    for index, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) == 3 and tokens[0] == '@' and tokens[1] in ('parameters', 'q_scaler'):
            end = index + 1 + int(tokens[2])
            lines[index + 1:end] = [
                f'{float(value) * factor:.7e}' for value in lines[index + 1:end]
            ]
    path.write_text('\n'.join(lines) + '\n')
    return path
