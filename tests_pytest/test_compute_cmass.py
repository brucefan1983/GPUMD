"""Compare the sampled center of mass with independently weighted XYZ frames."""

import subprocess

import numpy as np
import pytest
from ase.io import read


@pytest.mark.fast
def test_compute_cmass_matches_trajectory(tmp_path, gpumd_command):
    (tmp_path / 'lj.txt').write_text('lj 2 He Ar\n' + '1.0 1.0 2.0\n' * 4)
    (tmp_path / 'model.xyz').write_text(
        '4\n'
        'Lattice="10 0 0 0 10 0 0 0 10" pbc="T T T" '
        'Properties=species:S:1:pos:R:3:mass:R:1:group:I:1\n'
        'He 1 2 3 4 0\n'
        'He 1 7 8 5 1\n'
        'Ar 7 5 6 40 0\n'
        'Ar 6 7 1 41 1\n'
    )
    (tmp_path / 'run.in').write_text(
        'potential lj.txt\n'
        'velocity 300 seed 7\n'
        'time_step 2\n'
        'ensemble nve\n'
        'compute_cmass 2 group 0 0\n'
        'dump_xyz 2 dump.xyz unwrapped_position_only precision double\n'
        'run 5\n'
        'ensemble nve\n'
        'compute_cmass 2\n'
        'dump_xyz 2 dump.xyz unwrapped_position_only precision double\n'
        'run 4\n'
    )
    result = subprocess.run([gpumd_command], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    actual = np.loadtxt(tmp_path / 'cmass.out')
    frames = read(tmp_path / 'dump.xyz', index=':')
    assert actual.shape == (4, 4)
    assert len(frames) == 4
    np.testing.assert_allclose(actual[:, 0], [0.004, 0.008, 0.014, 0.018], rtol=0, atol=1e-12)

    expected = [np.average(frame.positions[[0, 2]], axis=0, weights=[4, 40]) for frame in frames[:2]]
    expected.extend(np.average(frame.positions, axis=0, weights=[4, 5, 40, 41]) for frame in frames[2:])
    np.testing.assert_allclose(actual[:, 1:], expected, rtol=0, atol=1e-9)


@pytest.mark.fast
def test_compute_cmass_tracks_periodic_crossing_without_dump(tmp_path, gpumd_command):
    (tmp_path / 'lj.txt').write_text('lj 1 Ar\n1.0 1.0 2.0\n')
    (tmp_path / 'model.xyz').write_text(
        '2\n'
        'Lattice="10 0 0 0 10 0 0 0 10" pbc="T T T" '
        'Properties=species:S:1:pos:R:3:vel:R:3:group:I:1\n'
        'Ar 9.75 1 1 0.1 0 0 0\n'
        'Ar 4 5 5 0 0 0 1\n'
    )
    (tmp_path / 'run.in').write_text(
        'potential lj.txt\n'
        'time_step 1\n'
        'ensemble nve\n'
        'compute_cmass 1 group 0 0\n'
        'run 3\n'
        'ensemble nve\n'
        'compute_cmass 1 group 0 0\n'
        'run 2\n'
    )
    result = subprocess.run([gpumd_command], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    steps = np.arange(1, 6)
    expected = np.column_stack((steps * 0.001, 9.75 + steps * 0.1, np.ones((5, 2))))
    np.testing.assert_allclose(np.loadtxt(tmp_path / 'cmass.out'), expected, rtol=0, atol=1e-9)
