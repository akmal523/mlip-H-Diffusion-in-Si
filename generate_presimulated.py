"""
generate_presimulated.py
========================
Generates synthetic but physically realistic MSD data for H diffusion
in Si.  The data is stored in  presimulated/  and allows running the
full analysis pipeline (analyze_msd.py) without a LAMMPS installation.

These files are included in the repository as pre-generated output
(presimulated/ is committed to git).  Re-run this script only if you
want to regenerate them with different parameters.

Physics basis
-------------
  Arrhenius parameters used:
    Ea  = 1.20 eV       (typical for H near-surface hopping in Si)
    D0  = 5e-3 cm²/s

  D(T) = D0 · exp(-Ea / kB·T)
  MSD(t) = 6·D·t   (3-D Einstein relation)

  Gaussian noise is added proportional to MSD to mimic single-atom
  statistical fluctuations.

The placeholder data is clearly labelled in the file header so it
cannot be confused with real simulation output.
"""

import os
import numpy as np

# =============================================================================
#  PARAMETERS
# =============================================================================

KB_EV       = 8.617333e-5  # Boltzmann constant [eV/K]
EA_EV       = 1.20         # activation energy  [eV]
D0_CM2S     = 5e-3         # pre-exponential    [cm²/s]
NOISE_LEVEL = 0.15         # relative noise amplitude (0.15 = 15%)
SEED        = 42

TEMPERATURES  = [700, 800, 1000, 1200, 1500]
TIMESTEPS     = {T: (0.001 if T <= 1000 else 0.0005) for T in TEMPERATURES}
N_POINTS      = {T: (7000  if T <= 1000 else 14000)  for T in TEMPERATURES}
OUTPUT_BASE   = "presimulated/Si64H1_box"

# =============================================================================
#  GENERATION
# =============================================================================

def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print("Generating placeholder MSD data")
    print(f"  Ea = {EA_EV} eV,  D0 = {D0_CM2S} cm²/s,  noise = {NOISE_LEVEL*100:.0f}%\n")

    for T in TEMPERATURES:
        dt = TIMESTEPS[T]
        n  = N_POINTS[T]

        folder = os.path.join(OUTPUT_BASE, f"T{T}K")
        os.makedirs(folder, exist_ok=True)

        D_cm2s = D0_CM2S * np.exp(-EA_EV / (KB_EV * T))
        D_A2ps = D_cm2s * 1e4   # 1 cm²/s = 1e4 Å²/ps

        steps = np.arange(1000, (n + 1) * 1000, 1000)
        times = steps * dt   # ps

        msd_mean  = 6.0 * D_A2ps * times
        noise_std = msd_mean * NOISE_LEVEL * (times[0] / times) ** 0.3
        msd_total = np.clip(msd_mean + rng.normal(0, noise_std), 0, None)

        msd_x = msd_total / 3.0 + rng.normal(0, msd_total * 0.05)
        msd_y = msd_total / 3.0 + rng.normal(0, msd_total * 0.05)
        msd_z = msd_total - msd_x - msd_y

        outpath = os.path.join(folder, f"msd_{T}K.dat")
        with open(outpath, "w") as fh:
            fh.write("# step  msd_total[A^2]  msd_x[A^2]  msd_y[A^2]  msd_z[A^2]\n")
            fh.write("# PLACEHOLDER — synthetic data for pipeline demonstration\n")
            fh.write(f"# Ea={EA_EV} eV  D0={D0_CM2S} cm2/s  seed={SEED}\n")
            fh.write(f"# D({T}K) = {D_cm2s:.4e} cm2/s\n")
            for s, mt, mx, my, mz in zip(steps, msd_total, msd_x, msd_y, msd_z):
                fh.write(f"{int(s)} {mt:.6f} {mx:.6f} {my:.6f} {mz:.6f}\n")

        print(f"  T = {T:5d} K  D = {D_cm2s:.3e} cm²/s  ({n} points)  -> {outpath}")

    print(f"\nDone.  Run the analysis with:")
    print(f"  python analyze_msd.py --presimulated")


if __name__ == "__main__":
    main()
