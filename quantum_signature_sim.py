#!/usr/bin/env python3
"""
quantum_signature_sim.py
========================
Monte Carlo simulation supporting Section V ("Performance Analysis") of
"Validity of Digital Signatures Under Section 3 IT Act 2000: Closing the Gap
of Quantum Cryptography in Indian Evidence Law" (S. D).

Question modelled
-----------------
Of the digitally signed electronic records that are still legally "live"
(i.e. could still be relied upon in a court proceeding), what share would
remain provable as authentic if RSA/ECC were broken (Q-Day)?

A record is SAFE at time t if EITHER
  (a) it was signed with a NIST post-quantum signature algorithm
      (ML-DSA / SLH-DSA), OR
  (b) it carries a qualified time-stamp dated strictly before t
      (time-stamp issued at signing, or an archival time-stamp applied later).
Otherwise it is EXPOSED: a genuine historical signature and a post-Q-Day
forgery would verify identically (the temporal-authenticity gap).

Policy scenarios (cumulative, mirroring Section IV of the paper)
---------------------------------------------------------------
  S0  Status quo                     - no legal change
  S1  Fix A                          - ML-DSA / SLH-DSA added to the technical
                                       standards licensed CAs follow (Sec. 18)
  S2  Fixes A + B                    - plus Sec. 3 time-stamp safe harbour
  S3  Fixes A + B + C                - plus binding regulatory mandate with
                                       deadline (default: December 2028)

Every assumption is a named constant or a sampled range below; change it and
re-run to test sensitivity. All results are reproducible with SEED.

Outputs (written to the working directory)
------------------------------------------
  fig3_problem_resolved.csv   % of live records safe, 2026-2035, per scenario
  fig4_exposure_at_qday.csv   mean and P90 % of live records exposed at Q-Day
  fig5_deadline_sensitivity.csv  residual exposure vs. mandate deadline (S3)
  summary.json                parameters, Monte Carlo standard errors, checks
"""

import csv
import json
import numpy as np

# ----------------------------------------------------------------------------
# Simulation controls
# ----------------------------------------------------------------------------
SEED = 2026
N_RUNS = 1000            # Monte Carlo runs
N_REC = 20_000           # signed records per run

# Records are created uniformly between these (decimal) years.
CREATE_START, CREATE_END = 2015.0, 2036.0

# Legal reliance horizon of a record (years) and the share of records in each
# class. Horizons follow Indian statutory periods:
#   3 y  - Limitation Act 1963, Sch., Art. 55 (breach of contract)
#   8 y  - Companies Act 2013, s.128(5) (books of account, 8 financial years)
#   12 y - Limitation Act 1963, Sch., Art. 65 (possession of immovable property)
# The SHARES are modelling assumptions.
HORIZONS = np.array([3.0, 8.0, 12.0])
HORIZON_SHARES = np.array([0.50, 0.30, 0.20])

# Q-Day (year RSA-2048 / ECC become breakable): triangular distribution.
# Mode = mid-2029 (estimate used in the paper, refs [9], [25]); long right tail
# reflects expert disagreement.
QDAY_MIN, QDAY_MODE, QDAY_MAX = 2027.5, 2029.5, 2038.0
QDAY_POINT = 2029.5      # single-point estimate used in Fig. 5

# Policy timing (from the paper / DST roadmap [10])
NOTIFICATION_YEAR = 2027.0   # Fix A: PQC signature algorithms recognised
TIMESTAMP_LAW_YEAR = 2028.0  # Fix B: Sec. 3 time-stamp amendment in force
MANDATE_START = 2027.0       # Fix C: mandate issued (financial-sector pilots)
DEFAULT_DEADLINE = 2029.0    # Fix C: "December 2028" deadline = start of 2029

LOGISTIC_K = 0.9             # adoption steepness (10%->90% in ~4.9 years)

YEARS = np.arange(2026, 2036)           # Fig. 3 x-axis (evaluated at year end)
DEADLINE_YEARS = np.arange(2027, 2034)  # Fig. 5 x-axis ("December <year>")

SCENARIOS = ["S0 Status quo", "S1 Fix A", "S2 Fixes A+B", "S3 Fixes A+B+C"]


def sample_parameters(rng):
    """Uncertain run-level parameters (uniform ranges = modelling assumptions)."""
    return {
        # organic PQC adoption midpoint without legal recognition
        "m0": rng.uniform(2032.0, 2035.0),
        # PQC adoption midpoint once algorithms are notified (Fix A)
        "m1": rng.uniform(2029.5, 2031.5),
        # share of records whose existing time-stamp stays independently
        # verifiable after Q-Day (status quo)
        "base_ts": rng.uniform(0.02, 0.08),
        # share of new records time-stamped once Fix B is law
        "ts_comp": rng.uniform(0.60, 0.85),
        # annual rate at which un-stamped archives are time-stamped (Fix B)
        "r_arch": rng.uniform(0.10, 0.25),
        # share of records issued by regulated entities (banks, CAs, CII)
        "rho": rng.uniform(0.55, 0.75),
        # compliance rate of regulated entities under a binding mandate
        "kappa": rng.uniform(0.85, 0.98),
        # Q-Day
        "qday": rng.triangular(QDAY_MIN, QDAY_MODE, QDAY_MAX),
    }


def sample_records(rng, p):
    """Record-level draws shared by all scenarios (common random numbers)."""
    c = rng.uniform(CREATE_START, CREATE_END, N_REC)
    h = rng.choice(HORIZONS, size=N_REC, p=HORIZON_SHARES)
    return {
        "c": c,                                  # creation (signing) time
        "e": c + h,                              # end of legal reliance
        "regulated": rng.random(N_REC) < p["rho"],
        "compliant": rng.random(N_REC) < p["kappa"],
        "u_pqc": rng.random(N_REC),
        "u_ts": rng.random(N_REC),
        "u_arch": rng.random(N_REC),
        "w_arch": rng.exponential(1.0, N_REC),
    }


def logistic(x, mid):
    return 1.0 / (1.0 + np.exp(-LOGISTIC_K * (x - mid)))


def scenario(name, r, p, deadline=DEFAULT_DEADLINE):
    """Return (pqc, tau): PQC-signed flag and earliest trusted time-stamp time
    (np.inf if none) for every record under the given scenario."""
    c = r["c"]
    inf = np.full(N_REC, np.inf)

    if name == SCENARIOS[0]:                                   # S0
        pqc = r["u_pqc"] < logistic(c, p["m0"])
        tau = np.where(r["u_ts"] < p["base_ts"], c, inf)
        return pqc, tau

    # Fix A: legal recognition accelerates adoption; it can never reduce the
    # organic (status quo) adoption, hence the maximum.
    a1 = np.maximum(logistic(c, p["m0"]),
                    logistic(c, p["m1"]) * (c >= NOTIFICATION_YEAR))
    if name == SCENARIOS[1]:                                   # S1
        pqc = r["u_pqc"] < a1
        tau = np.where(r["u_ts"] < p["base_ts"], c, inf)
        return pqc, tau

    # Fix B: time-stamp at signing after TIMESTAMP_LAW_YEAR, plus archival
    # time-stamping of un-stamped records at rate r_arch per year.
    ts_share = np.where(c >= TIMESTAMP_LAW_YEAR, p["ts_comp"], p["base_ts"])
    tau_sign = np.where(r["u_ts"] < ts_share, c, inf)
    tau_arch = np.maximum(c, TIMESTAMP_LAW_YEAR) + r["w_arch"] / p["r_arch"]
    pqc2 = r["u_pqc"] < a1
    tau2 = np.minimum(tau_sign, tau_arch)
    if name == SCENARIOS[2]:                                   # S2
        return pqc2, tau2

    # Fix C: compliant regulated entities ramp PQC signing linearly from
    # MANDATE_START to the deadline, time-stamp every new record from
    # MANDATE_START, and archive-stamp legacy records before the deadline.
    rc = r["regulated"] & r["compliant"]
    ramp = np.clip((c - MANDATE_START) / (deadline - MANDATE_START), 0.0, 1.0)
    pqc_rc = r["u_pqc"] < np.maximum(a1, ramp)
    tau_rc_sign = np.where(c >= MANDATE_START, c, inf)
    tau_rc_arch = np.maximum(c, MANDATE_START + r["u_arch"] * (deadline - MANDATE_START))
    tau_rc = np.minimum(np.minimum(tau_rc_sign, tau_rc_arch), tau2)
    pqc = np.where(rc, pqc_rc, pqc2)
    tau = np.where(rc, tau_rc, tau2)
    return pqc, tau


def share_safe(r, pqc, tau, t):
    """% of records live at time t that remain provable if Q-Day = t."""
    live = (r["c"] <= t) & (r["e"] > t)
    safe = live & (pqc | (tau < t))
    return 100.0 * safe.sum() / live.sum()


def main():
    rng = np.random.default_rng(SEED)
    n_s = len(SCENARIOS)
    resolved = np.zeros((N_RUNS, n_s, len(YEARS)))
    exposed_q = np.zeros((N_RUNS, n_s))
    dl_random = np.zeros((N_RUNS, len(DEADLINE_YEARS)))
    dl_point = np.zeros((N_RUNS, len(DEADLINE_YEARS)))
    qdays = np.zeros(N_RUNS)

    for k in range(N_RUNS):
        p = sample_parameters(rng)
        r = sample_records(rng, p)
        qdays[k] = p["qday"]
        for s, name in enumerate(SCENARIOS):
            pqc, tau = scenario(name, r, p)
            for j, y in enumerate(YEARS):
                resolved[k, s, j] = share_safe(r, pqc, tau, y + 1.0)
            exposed_q[k, s] = 100.0 - share_safe(r, pqc, tau, p["qday"])
        for j, y in enumerate(DEADLINE_YEARS):
            pqc, tau = scenario(SCENARIOS[3], r, p, deadline=y + 1.0)
            dl_random[k, j] = 100.0 - share_safe(r, pqc, tau, p["qday"])
            dl_point[k, j] = 100.0 - share_safe(r, pqc, tau, QDAY_POINT)

    se = lambda a, ax=0: a.std(axis=ax, ddof=1) / np.sqrt(a.shape[ax])

    # ---- Fig. 3 -----------------------------------------------------------
    res_mean = resolved.mean(axis=0)
    with open("fig3_problem_resolved.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Year"] + SCENARIOS)
        for j, y in enumerate(YEARS):
            w.writerow([int(y)] + [round(res_mean[s, j], 1) for s in range(n_s)])

    # ---- Fig. 4 -----------------------------------------------------------
    exp_mean = exposed_q.mean(axis=0)
    exp_p90 = np.percentile(exposed_q, 90, axis=0)
    with open("fig4_exposure_at_qday.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Scenario", "Mean exposed (%)", "P90 exposed (%)"])
        for s in range(n_s):
            w.writerow([SCENARIOS[s], round(exp_mean[s], 1), round(exp_p90[s], 1)])

    # ---- Fig. 5 -----------------------------------------------------------
    dl_r, dl_p = dl_random.mean(axis=0), dl_point.mean(axis=0)
    with open("fig5_deadline_sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Mandate deadline", "Exposed, Q-Day = mid-2029 (%)",
                    "Exposed, Q-Day uncertain (%)"])
        for j, y in enumerate(DEADLINE_YEARS):
            w.writerow([f"Dec {int(y)}", round(dl_p[j], 1), round(dl_r[j], 1)])

    # ---- Internal consistency checks --------------------------------------
    checks = {
        "fig3_values_in_0_100": bool(((resolved >= 0) & (resolved <= 100)).all()),
        "fig3_scenario_order_S0<=S1<=S2<=S3_every_year": bool(
            (np.diff(res_mean, axis=0) >= -1e-9).all()),
        "fig4_scenario_order_S0>=S1>=S2>=S3": bool(
            (np.diff(exp_mean) <= 1e-9).all()),
        "fig5_exposure_nondecreasing_with_later_deadline": bool(
            (np.diff(dl_r) >= -1e-9).all() and (np.diff(dl_p) >= -1e-9).all()),
    }
    summary = {
        "seed": SEED, "n_runs": N_RUNS, "n_records_per_run": N_REC,
        "qday_triangular": [QDAY_MIN, QDAY_MODE, QDAY_MAX],
        "qday_sample_mean": round(float(qdays.mean()), 2),
        "qday_sample_median": round(float(np.median(qdays)), 2),
        "fig3_max_mc_standard_error_pct": round(float(se(resolved).max()), 3),
        "fig4_max_mc_standard_error_pct": round(float(se(exposed_q).max()), 3),
        "fig5_max_mc_standard_error_pct": round(float(max(se(dl_random).max(),
                                                          se(dl_point).max())), 3),
        "checks": checks,
    }
    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
