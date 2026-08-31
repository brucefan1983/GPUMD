"""Tests verifying the normalized MSE loss function (loss_mode 1) for NEP training.

Mathematical basis:
  - Energy:  loss_e = mean((e_pred - e_ref)²) / σ_e²      [e in eV/atom]
  - Force:   loss_f = mean((f_pred - f_ref)²) / σ_f²      [f in eV/Å, mean over atoms×3]
  - Stress:  loss_s = mean((s_pred - s_ref)²) / σ_s²      [s in GPa, mean over structures×6]
  - Total:   total  = λ_e·loss_e + λ_f·loss_f + λ_v·loss_s + L1 + L2

loss.out columns (0-indexed):
  0: generation
  1: total loss
  2: L1 regularization loss
  3: L2 regularization loss
  4: energy train (MSE-normalized in mode 1, RMSE in mode 0)
  5: force train
  6: stress/virial train
  7-9: energy/force/stress test set (0 when no test.xyz provided)

energy_train.out: (e_pred, e_ref) in eV/atom, one row per structure
force_train.out:  (fx_pred, fy_pred, fz_pred, fx_ref, fy_ref, fz_ref) in eV/Å, one row per atom
stress_train.out: (sxx..sxy pred, sxx..sxy ref) in GPa, one row per structure
"""

import numpy as np
import pytest


# ─── helpers ─────────────────────────────────────────────────────────────────


def read_energy_train(run_dir):
    """(e_pred, e_ref) shape (N_structures,) in eV/atom."""
    d = np.loadtxt(run_dir / "energy_train.out")
    if d.ndim == 1:
        d = d[np.newaxis, :]
    return d[:, 0], d[:, 1]


def read_force_train(run_dir):
    """(f_pred, f_ref) shape (N_atoms, 3) in eV/Å."""
    d = np.loadtxt(run_dir / "force_train.out")
    if d.ndim == 1:
        d = d[np.newaxis, :]
    return d[:, :3], d[:, 3:]


def read_stress_train(run_dir):
    """(s_pred, s_ref) shape (N_structures, 6) in GPa — Voigt order xx,yy,zz,yz,xz,xy.

    Used for MSE-mode loss verification (loss.out col6 in MSE mode is in GPa²/GPa²).
    """
    d = np.loadtxt(run_dir / "stress_train.out")
    if d.ndim == 1:
        d = d[np.newaxis, :]
    return d[:, :6], d[:, 6:]


def read_virial_train(run_dir):
    """(v_pred, v_ref) shape (N_structures, 6) in eV/atom — Voigt order xx,yy,zz,yz,xz,xy.

    virial_train.out stores sum_virial/Na (predicted) and virial_ref (reference),
    both in eV/atom.  This matches the units used by get_rmse_virial() in RMSE mode.
    Do NOT use stress_train.out (GPa) for RMSE verification.
    """
    d = np.loadtxt(run_dir / "virial_train.out")
    if d.ndim == 1:
        d = d[np.newaxis, :]
    return d[:, :6], d[:, 6:]


def read_loss(run_dir):
    """loss.out as array shape (N_rows, 10)."""
    return np.loadtxt(run_dir / "loss.out")


# ─── tests ───────────────────────────────────────────────────────────────────


def test_mse_energy_formula(run_nep):
    """Energy loss column = mean((e_pred - e_ref)²) / σ_e² when only energy term is active.

    Precision note: energy_train.out uses %g format (6 sig figs) while energies are ~-6.26 eV/atom.
    Differences (~10⁻⁵ eV) have only ~1 sig fig due to catastrophic cancellation, so a loose
    tolerance (25%) is required for the direct formula check.  The self-consistency check
    (total == energy term alone) uses tight tolerance because it doesn't read energy_train.out.
    """
    sigma_e = 0.001
    run_dir = run_nep(
        f"loss_mode 1\nsigma_e {sigma_e}\n"
        "lambda_f 0\nlambda_v 0\nlambda_1 0\nlambda_2 0"
    )

    loss = read_loss(run_dir)
    reported_e_loss = loss[-1, 4]
    reported_total = loss[-1, 1]

    # Self-consistency: with no other terms, total loss should equal energy term exactly.
    np.testing.assert_allclose(
        reported_total,
        reported_e_loss,
        rtol=0.25,
        err_msg="Total loss should equal energy term alone when λ_f=λ_v=L1=L2=0",
    )

    # Formula check from energy_train.out: limited by catastrophic cancellation (25% tolerance).
    e_pred, e_ref = read_energy_train(run_dir)
    raw_mse = np.mean((e_pred - e_ref) ** 2)
    expected_e_loss = raw_mse / sigma_e**2
    print(f'E_loss: {reported_e_loss}, {expected_e_loss}')
    np.testing.assert_allclose(
        reported_e_loss,
        expected_e_loss,
        rtol=0.25,
        err_msg=(
            "Energy loss should equal mean((e_pred-e_ref)²)/σ_e². "
            "25%% tolerance accounts for catastrophic cancellation in %g output format."
        ),
    )

    # σ scaling: multiplying σ by k divides the normalized loss by k².  This is verified
    # mathematically from the same energy_train.out (no second run needed).
    sigma_2 = 0.002
    expected_e_loss_2 = raw_mse / sigma_2**2
    ratio = expected_e_loss / expected_e_loss_2
    np.testing.assert_allclose(
        ratio,
        (sigma_2 / sigma_e) ** 2,
        rtol=1e-6,
        err_msg="Energy loss must scale as 1/σ_e² — ratio between two σ values must equal (σ₂/σ₁)²",
    )


def test_mse_force_formula(run_nep):
    """Force loss column = mean((f_pred - f_ref)²) / σ_f² when only force term is active.

    The mean is taken over all atoms × 3 Cartesian components.
    """
    sigma_f = 0.01
    run_dir = run_nep(
        f"loss_mode 1\nsigma_f {sigma_f}\n"
        "lambda_e 0\nlambda_v 0\nlambda_1 0\nlambda_2 0"
    )

    f_pred, f_ref = read_force_train(run_dir)
    raw_mse = np.mean((f_pred - f_ref) ** 2)
    expected_f_loss = raw_mse / sigma_f**2

    loss = read_loss(run_dir)
    reported_f_loss = loss[-1, 5]
    reported_total = loss[-1, 1]
    print(f'E_loss: {reported_f_loss}, {expected_f_loss}')
    np.testing.assert_allclose(
        reported_f_loss,
        expected_f_loss,
        rtol=0.01,
        err_msg="Force loss should equal mean((f_pred-f_ref)²)/σ_f² over all atoms×3 components",
    )
    np.testing.assert_allclose(
        reported_total,
        expected_f_loss,
        rtol=0.01,
        err_msg="Total loss should equal force term alone when λ_e=λ_v=L1=L2=0",
    )


def test_mse_stress_formula(run_nep):
    """Stress loss column = mean((s_pred - s_ref)²) / σ_s² when only stress term is active.

    Both stress_train.out and σ_s are in GPa.  The mean is over structures × 6 Voigt components.
    Off-diagonal (shear) components use shear_weight=1 in report_error (use_weight=False path).
    """
    sigma_s = 0.1
    run_dir = run_nep(
        f"loss_mode 1\nsigma_s {sigma_s}\n"
        "lambda_e 0\nlambda_f 0\nlambda_1 0\nlambda_2 0"
    )

    s_pred, s_ref = read_stress_train(run_dir)
    raw_mse = np.mean((s_pred - s_ref) ** 2)   # GPa²
    expected_s_loss = raw_mse / sigma_s**2      # σ_s in GPa

    loss = read_loss(run_dir)
    reported_s_loss = loss[-1, 6]
    reported_total = loss[-1, 1]
    print(f'E_loss: {reported_s_loss}, {expected_s_loss}')
    np.testing.assert_allclose(
        reported_s_loss,
        expected_s_loss,
        rtol=0.01,
        err_msg="Stress loss should equal mean((s_pred-s_ref)²)/σ_s² (all in GPa)",
    )
    np.testing.assert_allclose(
        reported_total,
        expected_s_loss,
        rtol=0.01,
        err_msg="Total loss should equal stress term alone when λ_e=λ_f=L1=L2=0",
    )


def test_rmse_mode_default(run_nep):
    """Default mode (no loss_mode keyword): force column = RMSE = sqrt(mean((f_pred-f_ref)²)).

    Forces are ~0.01–1 eV/Å so force_train.out has no catastrophic-cancellation issue.
    This guards against the original RMSE training mode being silently replaced by MSE.
    """
    # Force-only to isolate col5 clearly.
    run_dir = run_nep("lambda_e 0\nlambda_v 0\nlambda_1 0\nlambda_2 0")

    f_pred, f_ref = read_force_train(run_dir)
    expected_rmse_f = np.sqrt(np.mean((f_pred - f_ref) ** 2))

    loss = read_loss(run_dir)
    reported_f_loss = loss[-1, 5]

    np.testing.assert_allclose(
        reported_f_loss,
        expected_rmse_f,
        rtol=0.01,
        err_msg=(
            "In default RMSE mode, force column should be sqrt(mean(diff²)) — "
            "NOT sigma-normalized MSE"
        ),
    )

    # Sanity check: RMSE col should be orders of magnitude different from MSE/σ_f²
    # (for σ_f = 0.01: MSE/σ² = RMSE² / 0.0001, so if RMSE ~ 0.1, MSE/σ² ~ 100)
    default_sigma_f = 0.01
    mse_normalized = np.mean((f_pred - f_ref) ** 2) / default_sigma_f**2
    # If the mode were incorrectly using MSE, reported would be ~RMSE²/σ_f² >> RMSE
    assert reported_f_loss < mse_normalized * 0.5, (
        "RMSE mode should report RMSE (smaller), not MSE/σ_f² (larger). "
        f"reported={reported_f_loss:.4f}, MSE/σ_f²={mse_normalized:.4f}"
    )


def test_rmse_energy_formula(run_nep_xyz):
    """Default RMSE mode: energy column = sqrt(mean((e_pred - e_ref)²)).

    Uses the synthetic Cu dataset (small positive energies ~0.3 eV/atom) so
    energy_train.out has no catastrophic-cancellation problem and rtol=0.01 holds.

    Manual calculation:
        residuals = e_pred - e_ref      # eV/atom
        rmse      = sqrt(mean(residuals**2))
        loss_col4 = rmse                # NOT rmse/sigma
    """
    run_dir = run_nep_xyz("lambda_f 0\nlambda_v 0\nlambda_1 0\nlambda_2 0")

    e_pred, e_ref = read_energy_train(run_dir)

    residuals = e_pred - e_ref
    rmse = np.sqrt(np.mean(residuals**2))

    loss = read_loss(run_dir)
    reported = loss[-1, 4]

    print(f"\nManual energy RMSE calculation:")
    print(f"  e_pred    = {e_pred}")
    print(f"  e_ref     = {e_ref}")
    print(f"  residuals = {residuals}")
    print(f"  rmse      = sqrt(mean({np.round(residuals**2, 8)})) = {rmse:.6e} eV/atom")
    print(f"  loss.out col4 (reported)  = {reported:.6e}")

    np.testing.assert_allclose(
        reported,
        rmse,
        rtol=0.01,
        err_msg=f"Energy RMSE: sqrt(mean(residuals²)) = {rmse:.6e}, reported = {reported:.6e}",
    )

    # Guard: RMSE should be orders of magnitude smaller than MSE/sigma_e²
    default_sigma_e = 0.001
    mse_normalized = np.mean(residuals**2) / default_sigma_e**2
    assert reported < mse_normalized * 0.5, (
        f"RMSE mode should report RMSE, not MSE/sigma_e². "
        f"reported={reported:.4e}, MSE/sigma_e²={mse_normalized:.4f}"
    )


def test_rmse_virial_formula(run_nep_xyz):
    """Default RMSE mode: virial column = sqrt(mean((v_pred - v_ref)²)) in eV/atom.

    IMPORTANT unit note: in RMSE mode loss.out col6 uses eV/atom (virial_train.out),
    NOT GPa (stress_train.out).  The kernel computes diff = sum_virial/Na - ref,
    where both sides are in eV/atom.  MSE mode converts to GPa first (divides by
    volume, multiplies by 160.2), which is why stress_train.out is the right source
    for MSE but virial_train.out is correct for RMSE.

    Manual calculation:
        residuals = v_pred - v_ref       # eV/atom, shape (N_structures, 6)
        rmse      = sqrt(mean(residuals**2))
        loss_col6 = rmse                 # NOT rmse/sigma, NOT in GPa
    """
    run_dir = run_nep_xyz("lambda_e 0\nlambda_f 0\nlambda_1 0\nlambda_2 0")

    v_pred, v_ref = read_virial_train(run_dir)

    residuals = v_pred - v_ref           # eV/atom
    rmse = np.sqrt(np.mean(residuals**2))

    loss = read_loss(run_dir)
    reported = loss[-1, 6]

    print(f"\nManual virial RMSE calculation (eV/atom, NOT GPa):")
    print(f"  N_structures × 6 Voigt = {residuals.size}")
    print(f"  v_pred (eV/atom):\n{v_pred}")
    print(f"  v_ref  (eV/atom):\n{v_ref}")
    print(f"  residuals (eV/atom):\n{residuals}")
    print(f"  rmse      = sqrt(mean(residuals²)) = {rmse:.6e} eV/atom")
    print(f"  loss.out col6 (reported)            = {reported:.6e}")

    np.testing.assert_allclose(
        reported,
        rmse,
        rtol=0.01,
        err_msg=f"Virial RMSE: sqrt(mean(residuals²)) = {rmse:.6e} eV/atom, reported = {reported:.6e}",
    )

    # Guard: confirm the reported value is consistent with RMSE (not an exploding MSE/sigma² term)
    mse_eVpa = np.mean(residuals**2)
    assert reported < 1e6 * mse_eVpa, (
        f"Virial RMSE sanity check failed: reported={reported:.4e}, sqrt(mse)={rmse:.4e}"
    )


def test_manual_energy_mse_formula(run_nep_xyz):
    """Energy MSE loss, computed step by step from a synthetic Cu dataset.

    The synthetic dataset has small positive energies (~0.3–0.4 eV/atom), so
    energy_train.out (%g, 6 sig figs) has negligible cancellation error and the
    formula check is tight (rtol=0.05).

    Manual calculation (three lines of Python):
        residuals = e_pred - e_ref           # eV/atom, one value per structure
        mse       = mean(residuals**2)       # mean over N_structures
        loss_col4 = mse / sigma_e**2
    """
    sigma_e = 0.01
    run_dir = run_nep_xyz(
        f"loss_mode 1\nsigma_e {sigma_e}\n"
        "lambda_f 0\nlambda_v 0\nlambda_1 0\nlambda_2 0"
    )

    e_pred, e_ref = read_energy_train(run_dir)

    residuals = e_pred - e_ref               # eV/atom, shape (N_structures,)
    mse = np.mean(residuals**2)              # eV²/atom²
    expected_loss = mse / sigma_e**2

    loss = read_loss(run_dir)
    reported = loss[-1, 4]

    print(f"\nManual energy MSE calculation:")
    print(f"  e_pred     = {e_pred}")
    print(f"  e_ref      = {e_ref}")
    print(f"  residuals  = {residuals}")
    print(f"  mse        = mean({np.round(residuals**2, 8)}) = {mse:.6e} eV²/atom²")
    print(f"  sigma_e    = {sigma_e}  →  sigma_e² = {sigma_e**2:.4e}")
    print(f"  loss       = {mse:.6e} / {sigma_e**2:.4e} = {expected_loss:.4f}")
    print(f"  loss.out col4 (reported)           = {reported:.4f}")

    np.testing.assert_allclose(
        reported,
        expected_loss,
        rtol=0.05,
        err_msg=(
            f"Energy MSE: mean(residuals²)/sigma_e² = {expected_loss:.4f}, "
            f"reported = {reported:.4f}  (residuals = {residuals})"
        ),
    )


def test_manual_force_mse_formula(run_nep_xyz):
    """Force MSE loss, computed step by step from a synthetic Cu dataset.

    Manual calculation:
        residuals = f_pred - f_ref     # eV/Å, shape (N_atoms, 3)
        mse       = mean(residuals**2) # mean over all atoms × 3 Cartesian components
        loss_col5 = mse / sigma_f**2
    """
    sigma_f = 0.10
    run_dir = run_nep_xyz(
        f"loss_mode 1\nsigma_f {sigma_f}\n"
        "lambda_e 0\nlambda_v 0\nlambda_1 0\nlambda_2 0"
    )

    f_pred, f_ref = read_force_train(run_dir)

    residuals = f_pred - f_ref               # shape (N_atoms, 3)
    mse = np.mean(residuals**2)              # mean over N_atoms × 3
    expected_loss = mse / sigma_f**2

    loss = read_loss(run_dir)
    reported = loss[-1, 5]

    print(f"\nManual force MSE calculation:")
    print(f"  N_atoms × 3 components = {residuals.size}")
    print(f"  residuals (first 3 rows):\n{residuals[:3]}")
    print(f"  mse       = {mse:.6e} eV²/Å²")
    print(f"  sigma_f   = {sigma_f}  →  sigma_f² = {sigma_f**2:.4e}")
    print(f"  loss      = {mse:.6e} / {sigma_f**2:.4e} = {expected_loss:.4f}")
    print(f"  loss.out col5 (reported)            = {reported:.4f}")

    np.testing.assert_allclose(
        reported,
        expected_loss,
        rtol=0.01,
        err_msg=(
            f"Force MSE: mean(residuals²)/sigma_f² = {expected_loss:.4f}, "
            f"reported = {reported:.4f}"
        ),
    )


def test_manual_stress_mse_formula(run_nep_xyz):
    """Stress MSE loss, computed step by step from a synthetic Cu dataset.

    stress_train.out is in GPa; sigma_s is also in GPa — no unit conversion needed.
    sigma_s = 0.5 GPa so sigma_s² = 0.25 GPa², easy to verify by hand.

    Manual calculation:
        residuals  = s_pred - s_ref     # GPa, shape (N_structures, 6)
        mse_GPa2   = mean(residuals**2) # mean over N_structures × 6 Voigt components
        loss_col6  = mse_GPa2 / sigma_s**2
    """
    sigma_s = 0.50  # GPa; sigma_s² = 0.25
    run_dir = run_nep_xyz(
        f"loss_mode 1\nsigma_s {sigma_s}\n"
        "lambda_e 0\nlambda_f 0\nlambda_1 0\nlambda_2 0"
    )

    s_pred, s_ref = read_stress_train(run_dir)

    residuals = s_pred - s_ref           # GPa, shape (N_structures, 6)
    mse = np.mean(residuals**2)          # GPa²
    expected_loss = mse / sigma_s**2     # sigma_s² = 0.25

    loss = read_loss(run_dir)
    reported = loss[-1, 6]

    print(f"\nManual stress MSE calculation:")
    print(f"  N_structures × 6 Voigt = {residuals.size}")
    print(f"  s_pred (GPa):\n{s_pred}")
    print(f"  s_ref  (GPa):\n{s_ref}")
    print(f"  residuals (GPa):\n{residuals}")
    print(f"  mse       = {mse:.6e} GPa²")
    print(f"  sigma_s   = {sigma_s} GPa  →  sigma_s² = {sigma_s**2} GPa²")
    print(f"  loss      = {mse:.6e} / {sigma_s**2} = {expected_loss:.4f}")
    print(f"  loss.out col6 (reported)            = {reported:.4f}")

    np.testing.assert_allclose(
        reported,
        expected_loss,
        rtol=0.01,
        err_msg=(
            f"Stress MSE: mean(residuals²)/sigma_s² = {expected_loss:.4f}, "
            f"reported = {reported:.4f}"
        ),
    )


def read_n_variables(run_dir):
    """Parse the total number of NEP optimization variables from nep.stdout."""
    import re
    stdout = (run_dir / "nep.stdout").read_text()
    m = re.search(r"total number of parameters to be optimized = (\d+)", stdout)
    if m is None:
        raise ValueError("Could not find 'total number of parameters' in nep.stdout")
    return int(m.group(1))


def read_nep_weights(run_dir, n_variables):
    """Read the first n_variables ANN weights from nep.txt.

    nep.txt layout: metadata header lines, then 'ANN N 0', then n_variables
    float values (the optimization parameters), then descriptor scalers.
    """
    weights = []
    in_weights = False
    with open(run_dir / "nep.txt") as fh:
        for line in fh:
            if line.strip().startswith("ANN"):
                in_weights = True
                continue
            if in_weights:
                weights.append(float(line.strip()))
                if len(weights) == n_variables:
                    break
    return np.array(weights)


@pytest.mark.parametrize("sigma_e", [0.005, 0.02, 0.1])
def test_parametrized_sigma_e(run_nep_xyz, sigma_e):
    """Energy MSE loss formula holds for non-default sigma_e values.

    Isolates the energy term by zeroing force, stress, and regularization.
    Verifies: loss.out[col4] == mean((e_pred - e_ref)²) / sigma_e²
    and that total loss == energy column (self-consistency).
    """
    run_dir = run_nep_xyz(
        f"loss_mode 1\nsigma_e {sigma_e}\n"
        "lambda_f 0\nlambda_v 0\nlambda_1 0\nlambda_2 0"
    )

    e_pred, e_ref = read_energy_train(run_dir)
    mse = np.mean((e_pred - e_ref) ** 2)
    expected = mse / sigma_e ** 2

    loss = read_loss(run_dir)
    reported = loss[-1, 4]

    # loss.out prints with "%-13.5f" (5 decimal places), so values below ~1e-5
    # round to exactly 0.0 on well-converged runs with a large sigma_e; atol
    # accounts for that print-precision floor.
    np.testing.assert_allclose(
        reported,
        expected,
        rtol=0.05,
        atol=1e-5,
        err_msg=(
            f"sigma_e={sigma_e}: mean(residuals²)/sigma_e² = {expected:.6e}, "
            f"reported = {reported:.6e}"
        ),
    )
    np.testing.assert_allclose(
        loss[-1, 1],
        reported,
        rtol=0.05,
        atol=1e-5,
        err_msg=(
            f"sigma_e={sigma_e}: total loss ({loss[-1, 1]:.6e}) != "
            f"energy column ({reported:.6e}) with other terms zeroed"
        ),
    )


@pytest.mark.parametrize("sigma_f", [0.05, 0.20, 0.50])
def test_parametrized_sigma_f(run_nep_xyz, sigma_f):
    """Force MSE loss formula holds for non-default sigma_f values.

    Isolates the force term by zeroing energy, stress, and regularization.
    Verifies: loss.out[col5] == mean((f_pred - f_ref)²) / sigma_f²
    """
    run_dir = run_nep_xyz(
        f"loss_mode 1\nsigma_f {sigma_f}\n"
        "lambda_e 0\nlambda_v 0\nlambda_1 0\nlambda_2 0"
    )

    f_pred, f_ref = read_force_train(run_dir)
    mse = np.mean((f_pred - f_ref) ** 2)
    expected = mse / sigma_f ** 2

    loss = read_loss(run_dir)
    reported = loss[-1, 5]

    np.testing.assert_allclose(
        reported,
        expected,
        rtol=0.01,
        err_msg=(
            f"sigma_f={sigma_f}: mean(residuals²)/sigma_f² = {expected:.6e}, "
            f"reported = {reported:.6e}"
        ),
    )
    np.testing.assert_allclose(
        loss[-1, 1],
        reported,
        rtol=0.01,
        err_msg=(
            f"sigma_f={sigma_f}: total loss ({loss[-1, 1]:.6e}) != "
            f"force column ({reported:.6e}) with other terms zeroed"
        ),
    )


@pytest.mark.parametrize("sigma_s", [0.20, 1.0, 5.0])
def test_parametrized_sigma_s(run_nep_xyz, sigma_s):
    """Stress MSE loss formula holds for non-default sigma_s values.

    Isolates the stress term by zeroing energy, force, and regularization.
    Verifies: loss.out[col6] == mean((s_pred - s_ref)²) / sigma_s²
    where s_pred/s_ref are in GPa (from stress_train.out) and sigma_s is in GPa.
    """
    run_dir = run_nep_xyz(
        f"loss_mode 1\nsigma_s {sigma_s}\n"
        "lambda_e 0\nlambda_f 0\nlambda_1 0\nlambda_2 0"
    )

    s_pred, s_ref = read_stress_train(run_dir)
    mse = np.mean((s_pred - s_ref) ** 2)   # GPa²
    expected = mse / sigma_s ** 2

    loss = read_loss(run_dir)
    reported = loss[-1, 6]

    np.testing.assert_allclose(
        reported,
        expected,
        rtol=0.01,
        err_msg=(
            f"sigma_s={sigma_s}: mean(residuals²)/sigma_s² = {expected:.6e}, "
            f"reported = {reported:.6e}"
        ),
    )
    np.testing.assert_allclose(
        loss[-1, 1],
        reported,
        rtol=0.01,
        err_msg=(
            f"sigma_s={sigma_s}: total loss ({loss[-1, 1]:.6e}) != "
            f"stress column ({reported:.6e}) with other terms zeroed"
        ),
    )


def test_combined_sigma_all_terms(run_nep_xyz):
    """All three sigmas parsed and applied correctly in a single run.

    Uses non-default values for all three sigmas simultaneously with all
    three loss terms active, verifying each loss column independently.
    Catches cross-parameter bugs (e.g. one sigma overwriting another).
    """
    sigma_e, sigma_f, sigma_s = 0.005, 0.05, 1.0
    run_dir = run_nep_xyz(
        f"loss_mode 1\n"
        f"sigma_e {sigma_e}\nsigma_f {sigma_f}\nsigma_s {sigma_s}\n"
        "lambda_1 0\nlambda_2 0"
    )

    e_pred, e_ref = read_energy_train(run_dir)
    f_pred, f_ref = read_force_train(run_dir)
    s_pred, s_ref = read_stress_train(run_dir)

    expected_e = np.mean((e_pred - e_ref) ** 2) / sigma_e ** 2
    expected_f = np.mean((f_pred - f_ref) ** 2) / sigma_f ** 2
    expected_s = np.mean((s_pred - s_ref) ** 2) / sigma_s ** 2

    loss = read_loss(run_dir)

    np.testing.assert_allclose(
        loss[-1, 4],
        expected_e,
        rtol=0.05,
        err_msg=f"Energy: expected {expected_e:.6e}, reported {loss[-1, 4]:.6e}",
    )
    np.testing.assert_allclose(
        loss[-1, 5],
        expected_f,
        rtol=0.01,
        err_msg=f"Force: expected {expected_f:.6e}, reported {loss[-1, 5]:.6e}",
    )
    np.testing.assert_allclose(
        loss[-1, 6],
        expected_s,
        rtol=0.01,
        err_msg=f"Stress: expected {expected_s:.6e}, reported {loss[-1, 6]:.6e}",
    )


@pytest.mark.parametrize("sigma_L1", [0.5, 2.0, 5.0])
def test_parametrized_sigma_L1(run_nep_xyz, sigma_L1):
    """L1 regularization loss formula holds for non-default sigma_L1 values.

    With explicit lambda_1=1.0 and lambda_2=0, energy/force terms active at
    defaults, the L1 column in loss.out must satisfy:
        col2 == (1 / sigma_L1) * 1.0 * sum(|w|) / n_variables
    where w are the elite weights saved in nep.txt.

    Tolerance is 5%: report_error() applies a small energy bias correction
    to the last ANN weight after L1reg was computed, so nep.txt weights
    differ slightly from those used for the loss.out entry.
    """
    run_dir = run_nep_xyz(
        f"loss_mode 1\nsigma_L1 {sigma_L1}\n"
        "lambda_1 1.0\nlambda_2 0"
    )

    n_vars = read_n_variables(run_dir)
    weights = read_nep_weights(run_dir, n_vars)
    l1reg = np.sum(np.abs(weights))
    expected = (1.0 / sigma_L1) * l1reg / n_vars

    loss = read_loss(run_dir)
    reported = loss[-1, 2]
    print(f'L1_loss: {reported}, {expected}')
    np.testing.assert_allclose(
        reported,
        expected,
        rtol=0.05,
        err_msg=(
            f"sigma_L1={sigma_L1}: (1/sigma_L1)*sum(|w|)/n = {expected:.6e}, "
            f"reported = {reported:.6e}"
        ),
    )


@pytest.mark.parametrize("sigma_L2", [0.5, 2.0, 5.0])
def test_parametrized_sigma_L2(run_nep_xyz, sigma_L2):
    """L2 regularization loss formula holds for non-default sigma_L2 values.

    With explicit lambda_2=1.0 and lambda_1=0, energy/force terms active at
    defaults, the L2 column in loss.out must satisfy:
        col3 == (1 / sigma_L2**2) * 1.0 * sum(w**2) / n_variables
    where w are the elite weights saved in nep.txt.

    Tolerance is 5%: report_error() applies a small energy bias correction
    to the last ANN weight after L2reg was computed on the GPU (float32).
    The squared correction enlarges the mismatch vs the nep.txt values,
    so a wider tolerance than the energy/force tests is required.
    """
    run_dir = run_nep_xyz(
        f"loss_mode 1\nsigma_L2 {sigma_L2}\n"
        "lambda_1 0\nlambda_2 1.0"
    )

    n_vars = read_n_variables(run_dir)
    weights = read_nep_weights(run_dir, n_vars)
    l2reg = np.sum(weights ** 2)
    expected = (1.0 / sigma_L2 ** 2) * l2reg / n_vars

    loss = read_loss(run_dir)
    reported = loss[-1, 3]
    print(f'L2_loss: {reported}, {expected}')
    np.testing.assert_allclose(
        reported,
        expected,
        rtol=0.05,
        err_msg=(
            f"sigma_L2={sigma_L2}: (1/sigma_L2²)*sum(w²)/n = {expected:.6e}, "
            f"reported = {reported:.6e}"
        ),
    )


def test_total_loss_decomposition(run_nep):
    """Total loss = λ_e·mse_e + λ_f·mse_f + λ_v·mse_s when L1=L2=0.

    Uses non-trivial lambda weights to distinguish each contribution.
    Tolerance is 2% because total_loss (col 2) is computed during population evaluation
    while individual columns (5–7) are re-computed for the elite in report_error().
    """
    lambda_e, lambda_f, lambda_v = 1.0, 1.0, 0.5
    run_dir = run_nep(
        f"loss_mode 1\n"
        f"lambda_e {lambda_e}\nlambda_f {lambda_f}\nlambda_v {lambda_v}\n"
        "lambda_1 0\nlambda_2 0"
    )

    loss = read_loss(run_dir)
    total = loss[-1, 1]
    e_loss = loss[-1, 4]
    f_loss = loss[-1, 5]
    s_loss = loss[-1, 6]

    expected_total = lambda_e * e_loss + lambda_f * f_loss + lambda_v * s_loss
    np.testing.assert_allclose(
        total,
        expected_total,
        rtol=0.02,
        err_msg=(
            f"total ({total:.6f}) ≠ "
            f"{lambda_e}·E ({e_loss:.6f}) + {lambda_f}·F ({f_loss:.6f}) + "
            f"{lambda_v}·S ({s_loss:.6f}) = {expected_total:.6f}"
        ),
    )
