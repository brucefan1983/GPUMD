#!/usr/bin/env python3
"""
Hydrogen Bond Analyzer for XYZ Trajectories
===========================================

This script analyzes hydrogen bonds (H-bonds) in molecular dynamics or quantum chemistry
trajectories provided in extended XYZ format. It supports both conventional and
unconventional H-bonds, including pi-H bonds (e.g., benzene rings, alkynes),
dihydrogen bonds, and metal-H bonds.

Theory
------
Hydrogen bonds are identified based on geometric criteria:
- The distance between donor and acceptor atoms (D...A) must be below a threshold (e.g., 3.5 Angstrom).
- The angle between Hydrogen-Donor-Acceptor must be sharp (e.g., < 30 Degree).
- The Hydrogen-Donor bond distance must be below a cutoff (e.g., < 1.2 Angstrom).

Unconventional H-bonds are also recognized by geometry:
- pi-H bonds: H atom interacts with the pi electron system (benzene ring, triple bond).
- Dihydrogen bonds: interaction between X-H and H-M (M = metal or B).
- Metal-H bonds: H atom interacts with a metal atom.

Features
--------
- Reads multi-frame .xyz trajectory files (supports periodic boundary conditions via Lattice=).
- Flexible donor and acceptor selection by index or element type.
- Automatic or manual detection of pi systems and metal centers.
- Parallel processing for speed.
- Outputs per-frame H-bond counts and detailed geometric statistics.

Usage
-----
Show all options:
    python hbond.py -h

Outputs
-------
- hbond_count.txt: Per-frame H-bond counts (regular and unconventional types)
- hbond_geom_all.txt: All H-bond geometries (frame, atom indices, distances, angles, type)
- Geometry distribution files: Donor-Hydrogen_distribution.txt, etc.

Author: chenzherui0124@foxmail.com
"""


import numpy as np
import re
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse
import networkx as nx

def parse_idx(s, n_atoms):
    if s is None:
        return list(range(n_atoms))
    idxs = set()
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            idxs.update(range(int(a)-1, int(b)))
        elif part:
            idxs.add(int(part)-1)
    return sorted(idxs)

def parse_frame(s, nframes):
    if s is None:
        return [i for i in range(nframes)]
    s = s.strip()
    if re.match(r"^\d+$", s):
        i = int(s)
        if i < 0 or i >= nframes:
            raise ValueError("Frame index out of range")
        return [i]
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if start > end:
            start, end = end, start
        start = max(start, 0)
        end = min(end, nframes - 1)
        return [i for i in range(start, end + 1)]
    raise ValueError("frame must be e.g. '0' or '0-1000'")

def parse_lattice(info_line):
    m = re.search(r'Lattice="([\d\.\s\-]+)"', info_line)
    if m:
        a = list(map(float, m.group(1).split()))
        box = np.array([a[0], a[4], a[8]])
        return box
    else:
        return None

def pbc_diff(r1, r2, box):
    delta = r1 - r2
    if box is not None:
        delta -= box * np.round(delta / box)
    return delta

def angle_between(v1, v2):
    dot = np.sum(v1*v2, axis=-1)
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cosang = dot / (n1 * n2 + 1e-12)
    cosang = np.clip(cosang, -1, 1)
    ang = np.arccos(cosang) * 180 / np.pi
    return ang

def parse_pi_list(s):
    if not s: return []
    pi_groups = []
    for block in s.split(','):
        block = block.strip()
        if '-' in block:
            a, b = block.split('-')
            pi_groups.append([i for i in range(int(a)-1, int(b))])
    return pi_groups

def get_transition_metals():
    return {'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
            'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
            'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg'}

def auto_detect_pi_centers(species, coords):
    """Detect six-membered all-carbon rings and triple bonds as pi centers."""
    cc_cut_ring = 1.7  # for rings
    cc_cut_triple = 1.25  # for triple bonds
    idx_c = [i for i, x in enumerate(species) if x == 'C']
    pi_groups = []

    # six-membered all-carbon rings
    if len(idx_c) >= 6:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(idx_c)
        for i, idxi in enumerate(idx_c):
            for j, idxj in enumerate(idx_c):
                if j <= i: continue
                r = np.linalg.norm(coords[idxi] - coords[idxj])
                if r < cc_cut_ring:
                    G.add_edge(idxi, idxj)
        cycles = nx.cycle_basis(G)
        for cyc in cycles:
            if len(cyc) == 6 and all(species[ii] == 'C' for ii in cyc):
                pi_groups.append(sorted(cyc))

    # triple bonds
    used = set()
    for i, idxi in enumerate(idx_c):
        for j, idxj in enumerate(idx_c):
            if j <= i: continue
            if (idxi, idxj) in used or (idxj, idxi) in used:
                continue
            r = np.linalg.norm(coords[idxi] - coords[idxj])
            if r < cc_cut_triple:
                
                neighbor_count_i = sum(
                    1 for k in idx_c if k != idxi and np.linalg.norm(coords[idxi] - coords[k]) < cc_cut_triple+0.1
                )
                neighbor_count_j = sum(
                    1 for k in idx_c if k != idxj and np.linalg.norm(coords[idxj] - coords[k]) < cc_cut_triple+0.1
                )
                if neighbor_count_i < 2 and neighbor_count_j < 2:
                    pi_groups.append([idxi, idxj])
                    used.add((idxi, idxj))
    return pi_groups

def auto_detect_metals(species):
    metals = get_transition_metals()
    found_metals = set([s for s in set(species) if s in metals])
    return sorted(found_metals)

def process_un(
    frame_idx, species, coords, box, pi_groups, metal_types, 
    R_cut, angle_cut, dH_cut, max_H, dt, at):

    records_pi, records_dih, records_metal = [], [], []
    hydrogen_indices = np.array([i for i, s in enumerate(species) if s == "H"])
    donor_indices = np.array([i for i, s in enumerate(species) if s in dt])

    # ---------- X-H...pi bond ----------
    if pi_groups:
        hydrogen_pos = coords[hydrogen_indices]
        donor_pos = coords[donor_indices]
        kdtree_h = cKDTree(hydrogen_pos, boxsize=box)
        for pi_atoms in pi_groups:
            if not pi_atoms: continue
            pi_center = np.mean(coords[pi_atoms], axis=0)
            for i_donor, donor_gidx in enumerate(donor_indices):
                donor_xyz = donor_pos[i_donor]
                h_dists, h_idxs = kdtree_h.query(donor_xyz, k=max_H, distance_upper_bound=dH_cut)
                for h_dist, h_local in zip(np.atleast_1d(h_dists), np.atleast_1d(h_idxs)):
                    if h_local >= len(hydrogen_pos): continue
                    h_gidx = hydrogen_indices[h_local]
                    h_xyz = hydrogen_pos[h_local]
                    R_vec = pbc_diff(pi_center, donor_xyz, box)
                    R = np.linalg.norm(R_vec)
                    r_vec = pbc_diff(pi_center, h_xyz, box)
                    r = np.linalg.norm(r_vec)
                    dh_vec = pbc_diff(h_xyz, donor_xyz, box)
                    angle = angle_between(dh_vec, R_vec)
                    if R <= R_cut and angle <= angle_cut:
                        records_pi.append((
                            frame_idx, donor_gidx+1, -1, h_gidx+1, h_dist, r, angle, R, 'pi'))

    # ---------- Dihydrogen bond: X-H...H-M (M=B or transition metal only) ----------
    metals = get_transition_metals()
    allowed_M_indices = np.array([i for i, s in enumerate(species) if (s == 'B') or (s in metals)])
    if allowed_M_indices.size > 0 and donor_indices.size > 0:
        acceptor_pos = coords[allowed_M_indices]
        donor_pos = coords[donor_indices]
        hydrogen_pos = coords[hydrogen_indices]
        kdtree_h = cKDTree(hydrogen_pos, boxsize=box)

        # Step 1: All X-H (dt) pairs (index in donor_indices + hydrogen_indices)
        XH_pairs = []
        for i_donor, donor_gidx in enumerate(donor_indices):
            donor_xyz = donor_pos[i_donor]
            h_dists, h_idxs = kdtree_h.query(donor_xyz, k=max_H, distance_upper_bound=dH_cut)
            for h_dist, h_local in zip(np.atleast_1d(h_dists), np.atleast_1d(h_idxs)):
                if h_local >= len(hydrogen_pos): continue
                h_gidx = hydrogen_indices[h_local]
                XH_pairs.append((donor_gidx, h_gidx, h_dist, donor_xyz, hydrogen_pos[h_local]))

        # Step 2: All M-H (at) pairs (index in allowed_M_indices + hydrogen_indices)
        MH_pairs = []
        for i_acc, acceptor_gidx in enumerate(allowed_M_indices):
            acceptor_xyz = acceptor_pos[i_acc]
            h_dists, h_idxs = kdtree_h.query(acceptor_xyz, k=max_H, distance_upper_bound=dH_cut)
            for h_dist, h_local in zip(np.atleast_1d(h_dists), np.atleast_1d(h_idxs)):
                if h_local >= len(hydrogen_pos): continue
                h_gidx = hydrogen_indices[h_local]
                MH_pairs.append((acceptor_gidx, h_gidx, h_dist, acceptor_xyz, hydrogen_pos[h_local]))

        # Fast lookup dict for MH (H as key), used for local search
        MH_h_to_M = {}
        for m_gidx, h_gidx, mh_dist, m_xyz, h_xyz in MH_pairs:
            if h_gidx not in MH_h_to_M:
                MH_h_to_M[h_gidx] = []
            MH_h_to_M[h_gidx].append((m_gidx, mh_dist, m_xyz))

        # KDTree for all MH-H atoms (acceptor Hs)
        MH_H_indices = np.array([h_gidx for (_, h_gidx, _, _, _) in MH_pairs])
        if len(MH_H_indices) > 0:
            MH_H_coords = np.array([h_xyz for (_, _, _, _, h_xyz) in MH_pairs])
            kdtree_MH_H = cKDTree(MH_H_coords, boxsize=box)
            for x_gidx, hX_gidx, xh_dist, x_xyz, hX_xyz in XH_pairs:
                neighbor_idxs = kdtree_MH_H.query_ball_point(hX_xyz, r=R_cut)
                for idx in neighbor_idxs:
                    hM_gidx = MH_H_indices[idx]
                    if hM_gidx == hX_gidx: continue
                    for m_gidx, mh_dist, m_xyz in MH_h_to_M[hM_gidx]:
                        hM_xyz = coords[hM_gidx]
                        R_vec = pbc_diff(hM_xyz, hX_xyz, box)
                        R = np.linalg.norm(R_vec)
                        dh_vec = pbc_diff(hX_xyz, x_xyz, box)
                        angle = angle_between(dh_vec, R_vec)
                        if R <= R_cut and angle <= angle_cut:
                            records_dih.append((
                                frame_idx, x_gidx+1, hM_gidx+1, hX_gidx+1, xh_dist, mh_dist, angle, R, 'dihydrogen'))
        # else: no MH_H atoms, records_dih remains empty

    # ---------- Metal-H bond: donor is from dt, acceptor is any metal ----------
    if metal_types:
        metal_indices = np.array([i for i, s in enumerate(species) if s in metal_types])
        if len(metal_indices) and donor_indices.size > 0:
            metal_pos = coords[metal_indices]
            donor_pos = coords[donor_indices]
            hydrogen_pos = coords[hydrogen_indices]
            kdtree_h = cKDTree(hydrogen_pos, boxsize=box)
            for i_donor, donor_gidx in enumerate(donor_indices):
                donor_xyz = donor_pos[i_donor]
                h_dists, h_idxs = kdtree_h.query(donor_xyz, k=max_H, distance_upper_bound=dH_cut)
                for h_dist, h_local in zip(np.atleast_1d(h_dists), np.atleast_1d(h_idxs)):
                    if h_local >= len(hydrogen_pos): continue
                    h_gidx = hydrogen_indices[h_local]
                    h_xyz = hydrogen_pos[h_local]
                    for i_met, met_gidx in enumerate(metal_indices):
                        met_xyz = metal_pos[i_met]
                        R_vec = pbc_diff(met_xyz, donor_xyz, box)
                        R = np.linalg.norm(R_vec)
                        r_vec = pbc_diff(met_xyz, h_xyz, box)
                        r = np.linalg.norm(r_vec)
                        dh_vec = pbc_diff(h_xyz, donor_xyz, box)
                        angle = angle_between(dh_vec, R_vec)
                        if R <= R_cut and angle <= angle_cut:
                            records_metal.append((
                                frame_idx, donor_gidx+1, met_gidx+1, h_gidx+1, h_dist, r, angle, R, 'metal'))
    return records_pi, records_dih, records_metal

def process_frame(args):
    (frame_idx, species, coords, box, donor_indices, acceptor_indices, R_cut, angle_cut, dH_cut, max_H, dt, at, un, pi_groups, metal_types) = args
    coords = coords % box if box is not None else coords

    donor_indices_type = [i for i in donor_indices if species[i] in dt]
    acceptor_indices_type = [i for i in acceptor_indices if species[i] in at]
    hydrogen_indices = [i for i, s in enumerate(species) if s == "H"]
    acceptor_pos = coords[acceptor_indices_type]
    donor_pos = coords[donor_indices_type]
    hydrogen_pos = coords[hydrogen_indices]
    kdtree_h = cKDTree(hydrogen_pos, boxsize=box)
    kdtree_acceptor = cKDTree(acceptor_pos, boxsize=box)
    hbond_count = 0
    geom_records_regular = []

    for i_donor, donor_gidx in enumerate(donor_indices_type):
        donor_xyz = donor_pos[i_donor]
        h_dists, h_idxs = kdtree_h.query(donor_xyz, k=max_H, distance_upper_bound=dH_cut)
        for h_dist, h_local in zip(np.atleast_1d(h_dists), np.atleast_1d(h_idxs)):
            if h_local >= len(hydrogen_pos): continue
            h_gidx = hydrogen_indices[h_local]
            h_xyz = hydrogen_pos[h_local]
            ac_idxs = kdtree_acceptor.query_ball_point(donor_xyz, r=R_cut)
            for j in ac_idxs:
                acceptor_gidx = acceptor_indices_type[j]
                if acceptor_gidx == donor_gidx: continue
                acceptor_xyz = acceptor_pos[j]
                R_vec = pbc_diff(acceptor_xyz, donor_xyz, box)
                R = np.linalg.norm(R_vec)
                r_vec = pbc_diff(acceptor_xyz, h_xyz, box)
                r = np.linalg.norm(r_vec)
                dh_vec = pbc_diff(h_xyz, donor_xyz, box)
                angle = angle_between(dh_vec, R_vec)
                if R <= R_cut and angle <= angle_cut:
                    hbond_count += 1
                    geom_records_regular.append((
                        frame_idx, donor_gidx+1, acceptor_gidx+1, h_gidx+1, h_dist, r, angle, R, 'regular'))

    records_pi, records_dih, records_metal = [], [], []
    if un.lower() in ['y', 'yes', 'true', '1']:
        records_pi, records_dih, records_metal = process_un(
            frame_idx, species, coords, box, pi_groups, metal_types,
            R_cut, angle_cut, dH_cut, max_H, dt, at)
    all_geom_records = geom_records_regular + records_pi + records_dih + records_metal
    return frame_idx, hbond_count, all_geom_records

def read_xyz_frames(filename):
    with open(filename) as f:
        frames = []
        while True:
            line = f.readline()
            if line == "":
                break
            try:
                n_atoms = int(line)
            except:
                break
            info = f.readline().strip()
            box = parse_lattice(info)
            atoms = []
            for _ in range(n_atoms):
                arr = f.readline().split()
                atoms.append(arr)
            species = [a[0] for a in atoms]
            coords = np.array([[float(a[1]), float(a[2]), float(a[3])] for a in atoms])
            frames.append( (len(frames), species, coords, box) )
        return frames

def save_hist(data, filename, bins=100, range=None):
    hist, bin_edges = np.histogram(data, bins=bins, range=range, density=True)
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    out = np.column_stack([centers, hist])
    np.savetxt(filename, out, header="Value\tProbability_density")

def analyze_geom_all(txtfile="hbond_geom_all.txt"):
    ddh_lst = []
    r_lst = []
    angle_lst = []
    R_lst = []
    with open(txtfile) as f:
        for line in f:
            if line.startswith("#") or line.strip().startswith("frame"): continue
            arr = line.split()
            if len(arr) < 8: continue
            ddh_lst.append(float(arr[4]))
            r_lst.append(float(arr[5]))
            angle_lst.append(float(arr[6]))
            R_lst.append(float(arr[7]))
    ddh_lst, r_lst, angle_lst, R_lst = map(np.array, [ddh_lst, r_lst, angle_lst, R_lst])
    for arr, fname, nbins, rrange in [
        (ddh_lst, "Donor-Hydrogen_distribution.txt", 100, (0, np.max(ddh_lst)*1.05 if len(ddh_lst) else 1)),
        (r_lst,   "Acceptor-Hydrogen_distribution.txt",   100, (0, np.max(r_lst)*1.05 if len(r_lst) else 1)),
        (angle_lst, "Acceptor-Donor-Hydrogen_angle_distribution.txt", 90, (0, 90)),
        (R_lst,   "Acceptor-Donor_distribution.txt",   100, (0, np.max(R_lst)*1.05 if len(R_lst) else 1))
    ]:
        if arr.size > 0:
            save_hist(arr, fname, bins=nbins, range=rrange)

def main():
    parser = argparse.ArgumentParser(description="Hydrogen bond analysis with optional unconventional (pi/dihydrogen/metal) H-bonds")
    parser.add_argument("xyzfile", type=str, help="Input xyz trajectory file")
    parser.add_argument("-di", type=str, default=None, help="Donor atom indices (1-based, default: all atoms)")
    parser.add_argument("-ai", type=str, default=None, help="Acceptor atom indices (1-based, default: all atoms)")
    parser.add_argument("-dt", type=str, default="O,N,F", help="Donor atom types (default: O,N,F)")
    parser.add_argument("-at", type=str, default="O,N,F", help="Acceptor atom types (default: O,N,F)")
    parser.add_argument("-R_cut", type=float, default=3.5, help="Donor-acceptor distance cutoff")
    parser.add_argument("-angle_cut", type=float, default=30.0, help="H-donor-acceptor angle cutoff")
    parser.add_argument("-dH_cut", type=float, default=1.2, help="Donor-H distance cutoff")
    parser.add_argument("-max_H", type=int, default=2, help="Max H per donor (2 for water, 1 for N-H etc)")
    parser.add_argument("-np", type=int, default=0, help="Number of processes (default: all)")
    parser.add_argument("-frame", type=str, default=None, help="Frame range, e.g. 0-1000 or 5 (default: all)")
    parser.add_argument("-un", type=str, default="n", help="Consider unconventional H-bonds (y/n, default n)")
    parser.add_argument("-pi_list", type=str, default="", help="List of pi-centers, e.g. 1001-1006,2001-2006")
    parser.add_argument("-metal_types", type=str, default="", help="Metal atom types for metal-H bonds, comma-separated")
    parser.add_argument("-analyze_geom", type=str, default="y", help="Analyze hbond_geom_all.txt? y/n (default: y)")
    args = parser.parse_args()

    frames = read_xyz_frames(args.xyzfile)
    nframes = len(frames)
    frame_indices = parse_frame(args.frame, nframes)
    frames_sel = [frames[i] for i in frame_indices]
    dt = [x.strip() for x in args.dt.split(",")]
    at = [x.strip() for x in args.at.split(",")]
    donor_indices = parse_idx(args.di, frames[0][2].shape[0])
    acceptor_indices = parse_idx(args.ai, frames[0][2].shape[0])

    # Only process pi_groups/metals if un is on
    if args.un.lower() in ['y', 'yes', 'true', '1']:
        if args.pi_list:
            pi_groups = parse_pi_list(args.pi_list)
        else:
            pi_groups = auto_detect_pi_centers(frames[0][1], frames[0][2])
            print(f"Auto-detected {len(pi_groups)} pi-centers (first frame):", [[i+1 for i in g] for g in pi_groups])
        if args.metal_types:
            metal_types = [x.strip() for x in args.metal_types.split(",") if x.strip()]
        else:
            metal_types = auto_detect_metals(frames[0][1])
            print(f"Auto-detected metal types (first frame):", metal_types)
    else:
        pi_groups = []
        metal_types = []

    args_list = [(idx, species, coords, box, donor_indices, acceptor_indices, args.R_cut, args.angle_cut, args.dH_cut,
                  args.max_H, dt, at, args.un, pi_groups, metal_types)
                 for (idx, species, coords, box) in frames_sel]

    results = []
    np = args.np if args.np > 0 else cpu_count()
    with Pool(np) as pool:
        for res in tqdm(pool.imap_unordered(process_frame, args_list), total=len(args_list)):
            results.append(res)
    results.sort()

    
    with open("hbond_count.txt", "w") as f:
        if args.un.lower() in ['y', 'yes', 'true', '1']:
            f.write("{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}\n".format(
                "frame", "regular", "di-hb", "pi-hb", "metal-hb"))
        else:
            f.write("{:<10s}{:<10s}\n".format("frame", "count"))
        for idx, count, records in results:
            if args.un.lower() in ['y', 'yes', 'true', '1']:
                reg = sum(1 for r in records if r[-1] == 'regular')
                dih = sum(1 for r in records if r[-1] == 'dihydrogen')
                pi  = sum(1 for r in records if r[-1] == 'pi')
                met = sum(1 for r in records if r[-1] == 'metal')
                f.write("{:<10d}{:<10d}{:<10d}{:<10d}{:<10d}\n".format(idx, reg, dih, pi, met))
            else:
                f.write("{:<10d}{:<10d}\n".format(idx, count))
  
    with open("hbond_geom_all.txt", "w") as f:
        f.write("{:>8s} {:>8s} {:>8s} {:>8s} {:>10s} {:>12s} {:>12s} {:>12s} {:>10s}\n".format(
            "frame","donor","acceptor","H","d_DH(A)","r(H-A)(A)","angle(deg)","R(D-A)(A)","type"))
        for _, _, records in results:
            for rec in records:
                f.write("{:8d} {:8d} {:8d} {:8d} {:10.5f} {:12.5f} {:12.5f} {:12.5f} {:>10s}\n".format(*rec))
    if args.analyze_geom.lower() in ['y', 'yes', 'true', '1']:
        analyze_geom_all("hbond_geom_all.txt")

if __name__ == "__main__":
    main()