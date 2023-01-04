from math import isclose

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False



def print_justified(val1, val2, close=True):
    print(f'{val1}'.ljust(30), end='')
    print(f'\t', end='')
    print(f'{val2}'.ljust(30), end='')
    if not close:
        print('!!!!')
    else:
        print('\n', end='')


def get_floats_in_file(file):
    entries = []
    with open(file, 'r') as f:
        for line in f.readlines():
            linem = line.strip()
            linem = linem.replace(' ', '=')
            linem = linem.replace(':', '=')
            linem = linem.replace('"', '=')
            split_line = linem.split('=')
            entries.extend([float(entry) for entry in split_line if isfloat(entry)])
    return entries


entries0 = get_floats_in_file('observer0.xyz')
entries1 = get_floats_in_file('observer1.xyz')
assert len(entries0) == len(entries1)
average_entries = [(entries0[i] + entries1[i])*0.5 for i in range(len(entries0))]
entries_to_test = get_floats_in_file('observer.xyz')
assert len(entries0) == len(entries_to_test)

n_total = len(entries_to_test)
n_close = 0
header_printed = False
for i in range(n_total):
    test = entries_to_test[i]
    ref = average_entries[i]
    close = isclose(test, ref, abs_tol=1e-4, rel_tol=1e-4)
    if not close:
        if not header_printed:
            print_justified('Test', 'Ref')
            header_printed = True
        print_justified(test, ref, close=close)
        raise ValueError(f"`observer.xyz` does not match average of `observer0.xyz` and `observer1.xyz` for value {i}")
    n_close += 1 if close else 0

print(f'{n_close}/{n_total} within tolerance')
print("Test passed")








