"""
analyze_msd.py
==============
Post-processing for H diffusion simulations in Si.

Reads the MSD output files written by LAMMPS (msd_{T}K.dat),
fits the linear (diffusive) regime, extracts diffusion coefficients D(T),
and builds an Arrhenius plot to obtain activation energy Ea and
pre-exponential factor D0.

Output (written to results/)
-----------------------------
  msd_{T}K.png          — MSD vs time with linear fit overlay
  arrhenius.png         — log10(D) vs 1000/T with fit line
  diffusion_summary.csv — D, R², fit range per temperature
  diffusion_summary.txt — human-readable results + Arrhenius parameters

Physics notes
-------------
Einstein relation (3-D):   MSD(t) = 6 D t
                            D = slope / 6       [Å²/ps]

Unit conversion:  1 Å²/ps  ×  10⁻⁴  =  1 cm²/s
    (1 Å² = 10⁻²⁰ m²,  1 ps = 10⁻¹² s
     → 10⁻²⁰/10⁻¹² = 10⁻⁸ m²/s = 10⁻⁴ cm²/s)

Arrhenius equation:  D(T) = D0 · exp(−Ea / kB T)
  → log10(D) = log10(D0) − [Ea / (kB · ln10)] · (1/T)

Fitting:  log10(D) vs  (1000/T)  gives
          slope_fit = −Ea · 1000 / (kB · ln10)
          → Ea = −slope_fit · kB · ln10 · 10⁻³    [eV]

Usage
-----
    python analyze_msd.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no display needed on HPC nodes
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =============================================================================
#  USER CONFIGURATION  (must match run_diffusion.py)
# =============================================================================

RUN_FOLDER   = "diffusion_runs"
STRUCT_NAME  = "Si64H1_box"
TEMPERATURES = [700, 800, 1000, 1200, 1500]   # [K]
OUTPUT_DIR   = "results"

# Fraction of the MSD trace used for the linear fit.
# 0.5 → use the last 50 % of the trajectory.
# Increase toward 1.0 for very long runs; decrease if MSD plateaus early.
FIT_FRACTION = 0.5

# Minimum R² accepted as a reliable D estimate (warn if below this).
MIN_R2 = 0.90

# Physical constants
KB_EV = 8.617333e-5   # Boltzmann constant [eV/K]

# =============================================================================
#  HELPERS
# =============================================================================

def read_msd_file(path: str) -> np.ndarray:
    """
    Read a LAMMPS 'fix print' MSD file.

    Expected columns: step  msd_total  msd_x  msd_y  msd_z  [Å²]
    Lines starting with '#' are skipped.

    Returns ndarray of shape (N, 5).  If the file only has 2 columns
    (legacy format: step msd_total) a zero-padded array is returned.
    """
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                row = [float(x) for x in parts]
            except ValueError:
                continue
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No data found in {path}")

    arr = np.array(rows)
    # Pad to 5 columns if only step + msd_total present
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n_cols = arr.shape[1]
    if n_cols < 5:
        pad = np.zeros((arr.shape[0], 5 - n_cols))
        arr = np.hstack([arr, pad])
    return arr


def read_timestep(infile: str, default: float = 0.001) -> float:
    """Extract the 'timestep' value [ps] from a LAMMPS input file."""
    if not os.path.isfile(infile):
        return default
    with open(infile, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("timestep"):
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        pass
    return default


def fit_linear(times: np.ndarray,
               msd:   np.ndarray,
               fit_fraction: float = 0.5
               ) -> tuple[float, float, float, float, float, float]:
    """
    Fit MSD = slope × t + intercept  over the last fit_fraction of the data.

    Returns
    -------
    slope, intercept, R², t_start, t_end, std_err_slope
    """
    n         = len(times)
    start_idx = max(0, int(n * (1.0 - fit_fraction)))
    if n - start_idx < 4:
        start_idx = 0    # fall back to full trace if fraction gives < 4 points

    t_fit   = times[start_idx:]
    msd_fit = msd[start_idx:]

    slope, intercept, r, _, stderr = linregress(t_fit, msd_fit)
    return slope, intercept, r ** 2, t_fit[0], t_fit[-1], stderr


def plot_msd(times:     np.ndarray,
             msd:       np.ndarray,
             slope:     float,
             intercept: float,
             t_start:   float,
             t_end:     float,
             D_cm2s:    float,
             R2:        float,
             T:         int,
             outpath:   str) -> None:
    """Save MSD vs time plot with linear-fit overlay."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(times, msd, lw=0.9, color="#2b7bba", alpha=0.9, label="MSD (LAMMPS)")

    t_line = np.array([t_start, t_end])
    ax.plot(
        t_line, slope * t_line + intercept,
        "r--", lw=2.0,
        label=f"Linear fit\nD = {D_cm2s:.3e} cm²/s\nR² = {R2:.4f}",
    )
    ax.axvspan(t_start, t_end, alpha=0.08, color="red", label="Fit region")

    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel("MSD (Å²)", fontsize=12)
    ax.set_title(f"H diffusion in Si  —  T = {T} K", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_arrhenius(inv_T:       np.ndarray,
                   log10D:      np.ndarray,
                   slope_A:     float,
                   intercept_A: float,
                   Ea_eV:       float,
                   D0_cm2s:     float,
                   R2_A:        float,
                   T_labels:    list[int],
                   outpath:     str) -> None:
    """Save Arrhenius plot: log10(D) vs 1000/T."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(inv_T, log10D, color="#2b7bba", s=70, zorder=3, label="Simulated D")

    x_pad = (inv_T.max() - inv_T.min()) * 0.08
    x_fit = np.linspace(inv_T.min() - x_pad, inv_T.max() + x_pad, 200)
    ax.plot(
        x_fit, slope_A * x_fit + intercept_A,
        "r-", lw=1.8,
        label=(
            f"Arrhenius fit\n"
            f"Eₐ = {Ea_eV:.3f} eV\n"
            f"D₀ = {D0_cm2s:.2e} cm²/s\n"
            f"R² = {R2_A:.4f}"
        ),
    )

    # Annotate temperature labels
    for x, y, T in zip(inv_T, log10D, T_labels):
        ax.annotate(
            f"{T} K", xy=(x, y),
            xytext=(4, 4), textcoords="offset points",
            fontsize=9, color="#444444",
        )

    ax.set_xlabel("1000 / T   (K⁻¹)", fontsize=12)
    ax.set_ylabel("log₁₀(D)   [D in cm²/s]", fontsize=12)
    ax.set_title("Arrhenius Plot  —  H diffusion in Si", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("H diffusion in Si  —  MSD analysis")
    print("=" * 60)

    records = []

    # ---- per-temperature MSD analysis ---------------------------------------
    for T in TEMPERATURES:
        run_dir  = os.path.join(RUN_FOLDER, STRUCT_NAME, f"T{T}K")
        msd_path = os.path.join(run_dir, f"msd_{T}K.dat")
        in_path  = os.path.join(run_dir, "in.diffusion")

        if not os.path.isfile(msd_path):
            print(f"  [SKIP] T={T} K — file not found: {msd_path}")
            continue

        dt  = read_timestep(in_path)   # [ps]
        raw = read_msd_file(msd_path)  # shape (N, 5)

        if raw.shape[0] < 20:
            print(
                f"  [SKIP] T={T} K — only {raw.shape[0]} data points; "
                "run may not have finished."
            )
            continue

        steps     = raw[:, 0]
        msd_total = raw[:, 1]   # [Å²]
        msd_x     = raw[:, 2]
        msd_y     = raw[:, 3]
        msd_z     = raw[:, 4]

        times = steps * dt   # [ps]

        # ---- 3-D diffusion coefficient (Einstein relation: MSD = 6Dt) -------
        slope, intercept, R2, t0, t1, slope_err = fit_linear(
            times, msd_total, FIT_FRACTION
        )

        if slope <= 0:
            print(f"  [WARN] T={T} K — fitted slope is non-positive ({slope:.3e}). "
                  "H atom may not be diffusing yet; try a longer run.")
            D_A2ps  = float("nan")
            D_cm2s  = float("nan")
        else:
            D_A2ps = slope / 6.0               # [Å²/ps]
            D_cm2s = D_A2ps * 1.0e-4           # [cm²/s]

        if R2 < MIN_R2 and not np.isnan(D_cm2s):
            print(f"  [WARN] T={T} K — R² = {R2:.3f} < {MIN_R2:.2f}; "
                  "linear regime may not be reached. Consider a longer run.")

        print(
            f"  T = {T:5d} K  |  D = {D_cm2s:.3e} cm²/s  |  "
            f"R² = {R2:.4f}  |  fit {t0:.0f}–{t1:.0f} ps"
        )

        records.append({
            "T_K":          T,
            "D_A2_per_ps":  D_A2ps,
            "D_cm2_per_s":  D_cm2s,
            "R2":           R2,
            "slope_A2_ps":  slope,
            "slope_stderr": slope_err,
            "fit_t_start_ps": t0,
            "fit_t_end_ps":   t1,
            "timestep_ps":  dt,
            "n_points":     raw.shape[0],
        })

        # ---- MSD plot --------------------------------------------------------
        if not np.isnan(D_cm2s):
            plot_msd(
                times, msd_total,
                slope, intercept, t0, t1,
                D_cm2s, R2, T,
                os.path.join(OUTPUT_DIR, f"msd_{T}K.png"),
            )

    # ---- save summary CSV ---------------------------------------------------
    if not records:
        print("\n[ERROR] No MSD data found. Run LAMMPS simulations first "
              "(python run_diffusion.py).")
        sys.exit(1)

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "diffusion_summary.csv")
    df.to_csv(csv_path, index=False, float_format="%.6e")

    # ---- Arrhenius analysis -------------------------------------------------
    valid = df["D_cm2_per_s"].notna() & (df["D_cm2_per_s"] > 0)
    df_v  = df[valid].copy()

    if len(df_v) < 2:
        print(
            "\n[WARN] Fewer than 2 valid D values — Arrhenius analysis skipped.\n"
            "Check warnings above and re-run with longer simulations."
        )
        return

    T_arr    = df_v["T_K"].values
    D_arr    = df_v["D_cm2_per_s"].values
    inv_T    = 1000.0 / T_arr         # [1000/K]
    log10D   = np.log10(D_arr)

    # log10(D) = log10(D0)  −  Ea/(kB · ln10) · (1/T)
    # slope_A  = d log10D / d(1000/T)  =  −Ea·1000 / (kB·ln10)
    slope_A, intercept_A, r_A, _, _ = linregress(inv_T, log10D)
    Ea_eV   = -slope_A * 1000.0 * KB_EV * np.log(10)   # [eV]
    D0_cm2s =  10 ** intercept_A                         # [cm²/s]
    R2_A    =  r_A ** 2

    print("\n" + "=" * 60)
    print("Arrhenius analysis")
    print("=" * 60)
    print(f"  Activation energy  Eₐ = {Ea_eV:.4f} eV")
    print(f"  Pre-exponential    D₀ = {D0_cm2s:.4e} cm²/s")
    print(f"  Arrhenius R²           = {R2_A:.4f}")

    plot_arrhenius(
        inv_T, log10D,
        slope_A, intercept_A,
        Ea_eV, D0_cm2s, R2_A,
        T_arr.tolist(),
        os.path.join(OUTPUT_DIR, "arrhenius.png"),
    )

    # ---- write text summary -------------------------------------------------
    lines = [
        "H Diffusion in Si — Analysis Summary",
        "=" * 45,
        "",
        "System  : 64 Si + 1 H atom",
        "Potential : GRACE-1L-OAM (machine-learning interatomic potential)",
        "Method  : Einstein relation, MSD = 6Dt  (3-D diffusion)",
        "Fit region : last {:.0%} of each trajectory".format(FIT_FRACTION),
        "",
        "Diffusion Coefficients",
        "-" * 45,
    ]
    for _, row in df.iterrows():
        if np.isnan(row["D_cm2_per_s"]):
            lines.append(f"  T = {row['T_K']:5.0f} K :  D = N/A   (fit failed)")
        else:
            lines.append(
                f"  T = {row['T_K']:5.0f} K :  D = {row['D_cm2_per_s']:.4e} cm²/s"
                f"   R² = {row['R2']:.4f}"
            )

    lines += [
        "",
        "Arrhenius Analysis",
        "-" * 45,
        f"  Activation energy  Eₐ  = {Ea_eV:.4f} eV",
        f"  Pre-exponential    D₀  = {D0_cm2s:.4e} cm²/s",
        f"  Arrhenius R²            = {R2_A:.4f}",
        "",
        "Unit Notes",
        "-" * 45,
        "  LAMMPS metal units: time in ps, length in Å.",
        "  MSD [Å²], timestep [ps] → D [Å²/ps] = slope/6",
        "  D [cm²/s] = D [Å²/ps] × 1e-4",
        "    (1 Å²/ps = 1e-8 m²/s = 1e-4 cm²/s)",
        "",
        "Output Files",
        "-" * 45,
        "  msd_{T}K.png           — MSD vs time, one plot per temperature",
        "  arrhenius.png          — Arrhenius plot",
        "  diffusion_summary.csv  — full numerical results table",
        "  diffusion_summary.txt  — this file",
    ]

    txt_path = os.path.join(OUTPUT_DIR, "diffusion_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nResults saved to  {OUTPUT_DIR}/")
    print(f"  {csv_path}")
    print(f"  {txt_path}")
    print(f"  {OUTPUT_DIR}/arrhenius.png")
    print(f"  {OUTPUT_DIR}/msd_*K.png")


if __name__ == "__main__":
    main()
