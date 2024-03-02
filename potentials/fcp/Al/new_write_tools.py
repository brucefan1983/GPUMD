import numpy as np
from itertools import product, permutations


def write_fcs_gpumd(fname_fc, fname_clusters, fcs, order, tol=1e-10):
    # cluster_lookup, fc_lookup = _get_lookup_data_smart(fcs, order, tol)
    cluster_lookup, fc_lookup = _get_lookup_data_naive(fcs, order, tol)
    _write_clusters(fname_clusters, cluster_lookup, order)
    _write_fc_lookup(fname_fc, fc_lookup, order)


def _write_fc_lookup(fname, fc_lookup, order):
    """ Writes the lookup force constants to file """
    fmt = '{}' + ' {}'*order
    with open(fname, 'w') as f:
        f.write(str(len(fc_lookup)) + '\n\n')
        for fc in fc_lookup:
            for xyz in product(range(3), repeat=order):
                f.write(fmt.format(*xyz, fc[xyz])+'\n')
            f.write('\n')


def _write_clusters(fname, cluster_lookup, order):
    """ Writes the cluster lookup to file """
    fmt = '{}' + ' {}'*order
    with open(fname, 'w') as f:
        f.write(str(len(cluster_lookup)) + '\n\n')
        for c, i in cluster_lookup.items():
            line = fmt.format(*c, i) + '\n'
            f.write(line)


def _get_clusters(fcs, order, tol):

    if order in [2, 3]:
        clusters = []
        for c in fcs._fc_dict.keys():
            if len(c) == order and np.linalg.norm(fcs[c]) > tol:
                for ci in permutations(c):
                    clusters.append(ci)
        clusters = list(sorted(set(clusters)))
    else:
        clusters = [c for c in fcs._fc_dict.keys() if len(c) == order and np.linalg.norm(fcs[c]) > tol]
    return clusters


def _get_lookup_data_naive(fcs, order, tol):
    """ Groups force constants for a given order into groups for which the
    force constant is identical. """
    fc_lookup = []
    cluster_lookup = dict()

    clusters = _get_clusters(fcs, order, tol)

    for c in clusters:
        fc1 = fcs[c]
        if np.linalg.norm(fc1) < tol:
            continue
        for i, fc2 in enumerate(fc_lookup):
            if np.linalg.norm(fc1 - fc2) < tol:
                cluster_lookup[c] = i
                break
        else:
            cluster_lookup[c] = len(fc_lookup)
            fc_lookup.append(fc1)
    return cluster_lookup, fc_lookup
