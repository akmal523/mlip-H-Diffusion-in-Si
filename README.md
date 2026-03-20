# H Diffusion in Si

Molecular dynamics study of hydrogen diffusion in a silicon slab using the **GRACE-1L-OAM** machine-learning interatomic potential.

The project covers the full workflow: structure preparation → automated multi-temperature NVT simulations → MSD analysis → Arrhenius activation energy extraction.

---

## System

| Property | Value |
|---|---|
| Atoms | 64 Si + 1 H |
| Si structure | Diamond cubic, *a* = 5.431 Å |
| Supercell | 2×2×2 unit cells |
| H position | Adsorbed on top Si surface |
| Boundary | Periodic x,y,z (vacuum gap in z prevents image interaction) |
| Ensemble | NVT (Nosé–Hoover thermostat) |
| Potential | GRACE-1L-OAM (Si–H) |

---

## Temperatures

700 K · 800 K · 1000 K · 1200 K · 1500 K

---

## Timestep

| T ≤ 1000 K | T > 1000 K |
|---|---|
| 0.001 ps | 0.0005 ps |

Shorter timestep at high temperature avoids integrator instability.

---

## Simulation Protocol

Each temperature run proceeds in three stages:

1. **Energy minimization** — conjugate-gradient, removes bad contacts from structure construction
2. **NVT equilibration** — 20 000 steps (20 ps at *dt* = 0.001 ps) to reach thermal equilibrium before recording MSD
3. **Production NVT** — records MSD every 1 000 steps

| Mode | Steps (T ≤ 1000 K) | Steps (T > 1000 K) |
|---|---|---|
| test | 2 000 | 2 000 |
| production | 7 000 000 | 14 000 000 |

---

## Diffusion Analysis

Diffusion coefficient from the **Einstein relation** (3-D):

$$D = \frac{1}{6} \frac{d \langle |\mathbf{r}(t) - \mathbf{r}(0)|^2 \rangle}{dt}$$

Activation energy from the **Arrhenius equation**:

$$D(T) = D_0 \exp\!\left(-\frac{E_a}{k_B T}\right)$$

Fit performed on log₁₀(D) vs 1000/T.

---

## Project Structure

```
H-diffusion-in-Si/
├── si_with_h.lmp          # LAMMPS structure file (64 Si + 1 H)
├── in.diffusion.lammps    # annotated LAMMPS input template
├── run_diffusion.py       # launches multi-temperature simulations
├── analyze_msd.py         # MSD fitting + Arrhenius analysis
├── grace.yml              # conda environment
└── results/
    └── README_results.txt # description of all output files
```

---

## Setup

### 1. Conda environment

```bash
conda env create -f grace.yml
conda activate lammps_grace_env
```

### 2. LAMMPS + GRACE

LAMMPS must be compiled with the GRACE pair style. See the [GRACE documentation](https://github.com/ICAMS/grace-tensorpotential) for build instructions.

Update the paths in `run_diffusion.py`:

```python
LAMMPS_EXE = "/path/to/lammps/build/lmp"
GRACE_PATH = "/path/to/GRACE-1L-OAM"
```

---

## Usage

```bash
# Quick test (2 000 steps)
python run_diffusion.py          # default mode = "test"

# Monitor
tail -f diffusion_runs/Si64H1_box/T700K/log.lammps

# After all jobs finish
python analyze_msd.py
```

Results are written to `results/`.

---

## Output

| File | Description |
|---|---|
| `diffusion_runs/.../log.lammps` | LAMMPS thermo output per run |
| `diffusion_runs/.../msd_{T}K.dat` | MSD time series (Å²) |
| `diffusion_runs/.../dump.atom` | Trajectory with wrapped + unwrapped coords |
| `results/msd_{T}K.png` | MSD plot with linear fit |
| `results/arrhenius.png` | Arrhenius plot |
| `results/diffusion_summary.csv` | D, R², fit range per temperature |
| `results/diffusion_summary.txt` | Activation energy, D₀, unit notes |

See `results/README_results.txt` for a detailed description of every output file.

---

## Key Corrections vs. First Draft

| Bug | Fix |
|---|---|
| `velocity` placed before `minimize` | Moved after `reset_timestep 0` |
| `compute msd com yes` on 1-atom group → MSD = 0 | Changed to `com no` |
| Pressure: kinetic energy subtracted twice | Removed; `stress/atom NULL` already includes kinetic contribution |
| `z = parts[-1]` breaks with image flags | Fixed to `parts[4]` (atomic style, column 4 is always z) |
| No equilibration before recording MSD | Added 20 ps NVT equilibration stage |
| GRACE path hardcoded per user | Configurable `GRACE_PATH` variable |

---

## References

- Thompson et al., *LAMMPS — A flexible simulation tool for particle-based materials modeling*, Comp. Phys. Comm. **271**, 108171 (2022)
- Kovács et al., *MACE: Fast and accurate machine learning force fields with higher order equivariant message passing*, NeurIPS 2022
- GRACE potential: <https://github.com/ICAMS/grace-tensorpotential>
