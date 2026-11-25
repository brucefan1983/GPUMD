"""Compare GPUMD's public MEAM interface against a LAMMPS meam executable."""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


CASES = {
    "Fe": ("Fe.meam", ("Fe",), 2.8285, ((0, 0, 0), (0.5, 0.5, 0.5))),
    "Ge": (
        "Ge.meam", ("Ge",), 5.6580326381,
        ((0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0),
         (0.25, 0.25, 0.25), (0.25, 0.75, 0.75), (0.75, 0.25, 0.75), (0.75, 0.75, 0.25)),
    ),
    "Ga_In": ("GaIn.meam", ("Ga", "In"), 5.6552, ((0, 0, 0), (0.5, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0.5))),
    "Al_Co_Fe": ("FeCoAl.meam", ("Fe", "Co", "Al"), 5.15,
                 ((0, 0, 0), (0.5, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0.5))),
}


def execute(executable, cwd, log_name, *args):
    with (cwd / log_name).open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [str(executable), *args], cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
            timeout=90, check=False,
        )
    if result.returncode:
        raise RuntimeError((cwd / log_name).read_text(encoding="utf-8", errors="replace"))


def write_gpumd(cwd, symbols, types, positions, length, periodic, steps, dt):
    lattice = " ".join(f"{i:.17g}" for i in (np.eye(3) * length).ravel())
    pbc = "T T T" if periodic else "F F F"
    header = f'{len(types)}\nLattice="{lattice}" pbc="{pbc}" Properties=species:S:1:pos:R:3:vel:R:3\n'
    rows = [
        f"{symbols[t]} {' '.join(f'{x:.17g}' for x in xyz)} 0 0 0\n"
        for t, xyz in zip(types, positions)
    ]
    (cwd / "model.xyz").write_text(header + "".join(rows), encoding="ascii")
    thermo = "dump_thermo 1\n" if steps > 1 else ""
    (cwd / "run.in").write_text(
        f"potential meam.txt\ntime_step {dt}\nensemble nve\n"
        f"{thermo}dump_xyz {steps} gpumd.xyz precision double force potential virial\nrun {steps}\n",
        encoding="ascii",
    )
    (cwd / "gpumd.xyz").unlink(missing_ok=True)
    (cwd / "thermo.out").unlink(missing_ok=True)


def gpumd_results(executable, cwd, symbols, types, positions, length, periodic, steps=1, dt=1e-5):
    write_gpumd(cwd, symbols, types, positions, length, periodic, steps, dt)
    execute(executable, cwd, "gpumd.log")
    rows = np.loadtxt(cwd / "gpumd.xyz", skiprows=2, usecols=range(1, 17), ndmin=2)
    return rows[:, 3:6], rows[:, 6], rows[:, 7:].reshape(-1, 3, 3).sum(axis=0)


def lammps_results(executable, cwd, symbols, types, positions, length, periodic):
    header = (
        f"MEAM validation\n\n{len(types)} atoms\n{len(symbols)} atom types\n"
        f"0 {length:.17g} xlo xhi\n0 {length:.17g} ylo yhi\n0 {length:.17g} zlo zhi\n\nAtoms # atomic\n\n"
    )
    rows = [
        f"{i} {t + 1} {' '.join(f'{x:.17g}' for x in xyz)}\n"
        for i, (t, xyz) in enumerate(zip(types, positions), 1)
    ]
    (cwd / "system.data").write_text(header + "".join(rows), encoding="ascii")
    descriptor = (cwd / "meam.txt").read_text(encoding="ascii").splitlines()
    library, params = descriptor[1:3]
    names = " ".join(symbols) if len(descriptor) == 3 else " ".join(descriptor[3].split()[1:])
    boundary = "p p p" if periodic else "f f f"
    (cwd / "in.lmp").write_text(
        f"units metal\natom_style atomic\nboundary {boundary}\nread_data system.data\n"
        f"mass * 1\npair_style meam\npair_coeff * * {library} {names} {params} {names}\n"
        "neighbor 2.0 bin\n"
        "compute e all pe/atom\ncompute w all pressure NULL virial\n"
        "thermo_style custom step pe c_w[1] c_w[2] c_w[3] c_w[4] c_w[5] c_w[6]\n"
        "thermo_modify format float %.17g\nrun 0\n"
        'write_dump all custom lammps.dump id fx fy fz c_e modify sort id format float "%.17g"\n'
        'print "$(c_w[1]:%.17g) $(c_w[2]:%.17g) $(c_w[3]:%.17g) $(c_w[4]:%.17g) '
        '$(c_w[5]:%.17g) $(c_w[6]:%.17g)" file virial.dat screen no\n',
        encoding="ascii",
    )
    execute(executable, cwd, "lammps.log", "-in", "in.lmp", "-log", "none")
    data = np.loadtxt(cwd / "lammps.dump", skiprows=9, ndmin=2)
    np.testing.assert_array_equal(data[:, 0], np.arange(1, len(types) + 1))
    xx, yy, zz, xy, xz, yz = np.loadtxt(cwd / "virial.dat") * length**3 / 1602176.5
    return data[:, 1:4], data[:, 4], np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def evaluate(gpumd, lammps, cwd, label, symbols, types, positions, length, periodic):
    actual = gpumd_results(gpumd, cwd, symbols, types, positions, length, periodic)
    expected = lammps_results(lammps, cwd, symbols, types, positions, length, periodic)
    error = [float(np.max(np.abs(a - b))) for a, b in zip(actual, expected)]
    print(f"{label}: max |dF|={error[0]:.3g} eV/A, |dE_atom|={error[1]:.3g} eV, "
          f"|dW|={error[2]:.3g} eV", flush=True)
    np.testing.assert_allclose(actual[0], expected[0], atol=2e-5, rtol=1e-6)
    np.testing.assert_allclose(actual[1], expected[1], atol=2e-6, rtol=1e-6)
    np.testing.assert_allclose(actual[2], expected[2], atol=5e-4, rtol=1e-6)
    np.testing.assert_allclose(actual[0].sum(axis=0), 0.0, atol=1e-8)

    if not periodic:
        shifted = positions.copy()
        displacement = 2e-5
        shifted[2, 1] += displacement
        plus = gpumd_results(gpumd, cwd, symbols, types, shifted, length, False)[1].sum()
        shifted[2, 1] -= 2 * displacement
        minus = gpumd_results(gpumd, cwd, symbols, types, shifted, length, False)[1].sum()
        np.testing.assert_allclose(actual[0][2, 1], -(plus - minus) / (2 * displacement), atol=3e-6)


def test_case(gpumd, lammps, source, cwd, case):
    parameter_file, symbols, lattice, basis = CASES[case]
    shutil.copyfile(source / "library.meam", cwd / "library.meam")
    shutil.copyfile(source / parameter_file, cwd / parameter_file)
    (cwd / "meam.txt").write_text(
        f"meam {len(symbols)} {' '.join(symbols)}\nlibrary.meam\n{parameter_file}\n",
        encoding="ascii",
    )
    rng = np.random.default_rng(104)
    p = np.array(basis)
    repeats = 4
    positions = (
        np.indices((repeats,) * 3).reshape(3, -1).T[:, None, :] + p[None, :, :]
    ).reshape(-1, 3) * lattice
    types = np.arange(len(positions)) % len(symbols)
    positions += rng.normal(0.0, 0.022, positions.shape)
    length = repeats * lattice
    positions %= length
    evaluate(gpumd, lammps, cwd, f"{case} periodic", symbols, types, positions, length, True)
    drifts = []
    for steps, dt in ((24, 0.2), (192, 0.025)):
        gpumd_results(gpumd, cwd, symbols, types, positions, length, True, steps=steps, dt=dt)
        total = np.loadtxt(cwd / "thermo.out", usecols=(1, 2), ndmin=2).sum(axis=1)
        assert len(total) == steps
        drifts.append(np.ptp(total) / len(types))
    coarse, fine = drifts
    print(f"{case} NVE: energy range {coarse:.3g} and {fine:.3g} eV/atom "
          "at 0.2 and 0.025 fs", flush=True)
    assert fine < 2e-6 and (coarse < 2e-6 or fine < coarse / 2)
    xyz = np.array([[8, 8, 8], [10.61, 8, 8], [9.35, 9.79, 8.11], [9.25, 7.65, 10.33]])
    evaluate(gpumd, lammps, cwd, f"{case} cluster", symbols, np.arange(4) % len(symbols),
             xyz, 18.0, False)


def test_reference_lattices(gpumd, lammps, root):
    symbols = ("Al", "Fe")
    xyz = np.array([[8, 8, 8], [10.61, 8, 8], [9.35, 9.79, 8.11], [9.25, 7.65, 10.33]])
    lattices = ("fcc", "bcc", "hcp", "dia", "dia3", "dim", "sc", "b1", "b2", "c11",
                "l12", "ch4", "lin", "zig", "tri")
    density_forms = (3, 0, 1, 4, -5)
    for index, name in enumerate(lattices):
        cwd = root / name
        cwd.mkdir()
        first_form = density_forms[index % len(density_forms)]
        second_form = density_forms[(index + 1) % len(density_forms)]
        (cwd / "library.meam").write_text(
            f"'Al_example' 'dia' 4 13 26.98 4.4 2.1 2.3 2.2 2.1 5.4 3.4 0.9 1 1.1 1.2 1.3 1 {first_form}\n"
            f"'Fe_example' 'bcc' 8 26 55.85 4.7 2.5 2.1 2.4 2.2 2.9 4.3 0.6 1 1.2 1.3 1.4 1 {second_form}\n",
            encoding="ascii",
        )
        (cwd / "parameters.meam").write_text(
            f"lattce(1,2)='{name}'\nre(1,2)=2.5\nEc(1,2)=3.5\nalpha(1,2)=4.5\n"
            f"nn2(1,2)={int(name != 'c11')}\nialloy={index % 3}\n"
            f"erose_form={index % 3}\nmixture_ref_t={int(index % 4 == 1)}\n"
            f"bkgd_dyn={int(index % 4 == 2)}\nemb_lin_neg={int(index % 4 == 3)}\n"
            "theta(1,2)=117\nrc=5.0\nCmin(1,1,2)=0.8\nCmin(2,2,1)=0.8\n",
            encoding="ascii",
        )
        (cwd / "meam.txt").write_text(
            "meam 2 Al Fe\nlibrary.meam\nparameters.meam\n"
            "library_elements Al_example Fe_example\n", encoding="ascii",
        )
        evaluate(gpumd, lammps, cwd, name, symbols, np.arange(4) % 2, xyz, 18.0, False)


def test_seven_elements(gpumd, lammps, root, source):
    cwd = root / "seven"
    cwd.mkdir()
    library, params = "library_2nn.meam", "MoCoNiVFeAlCr_2nn.meam"
    for filename in (library, params):
        shutil.copyfile(source / filename, cwd / filename)
    symbols = ("Mo", "Co", "Ni", "V", "Fe", "Al", "Cr")
    (cwd / "meam.txt").write_text(
        f"meam 7 {' '.join(symbols)}\n{library}\n{params}\n", encoding="ascii",
    )
    rng = np.random.default_rng(204)
    length = 17.7
    positions = np.indices((6,) * 3).reshape(3, -1).T * (length / 6)
    positions = (positions + rng.normal(0.0, 0.025, positions.shape)) % length
    evaluate(gpumd, lammps, cwd, "seven-element alloy", symbols, np.arange(len(positions)) % 7,
             positions, length, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpumd", type=Path, required=True)
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--potentials", type=Path)
    parser.add_argument("--lammps-potentials", type=Path)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="meam-", dir=args.work_dir) as folder:
        gpumd = args.gpumd.resolve()
        lammps = args.lammps.resolve()
        root = Path(folder)
        test_reference_lattices(gpumd, lammps, root)
        if args.potentials:
            for case in CASES:
                cwd = root / case
                cwd.mkdir()
                test_case(gpumd, lammps, args.potentials / case, cwd, case)
        if args.lammps_potentials:
            test_seven_elements(gpumd, lammps, root, args.lammps_potentials)


if __name__ == "__main__":
    main()
