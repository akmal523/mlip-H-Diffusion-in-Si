"""
run_diffusion.py
================
Launches LAMMPS NVT diffusion simulations of 1 H atom in a 64-Si slab
at multiple temperatures using the GRACE-1L-OAM ML interatomic potential.

For each temperature a separate directory is created, an in.diffusion
input file is generated with the correct parameters, and LAMMPS is
launched in the background (nohup + mpirun).

Usage
-----
    python run_diffusion.py

Set mode = "test" for short verification runs,
    mode = "production" for full diffusion statistics.

After jobs finish, run:
    python analyze_msd.py
"""

import os
import subprocess
import shlex

# =============================================================================
#  USER CONFIGURATION
# =============================================================================

STRUCT_FILE  = "si_with_h.lmp"       # LAMMPS data file (atom_style atomic)
STRUCT_NAME  = "Si64H1_box"          # label used for output folders
TEMPERATURES = [700, 800, 1000, 1200, 1500]   # [K]
RUN_FOLDER   = "diffusion_runs"

# Path to the compiled LAMMPS executable
LAMMPS_EXE   = os.path.expanduser("~/MarquesN/lammps/build/lmp")

# Path to the GRACE potential directory (contains element files)
GRACE_PATH   = os.path.expanduser("~/.cache/grace/GRACE-1L-OAM")

# "test"       : 2 000 steps,  thermo every 100  — quick sanity check
# "production" : 7 M / 14 M steps, thermo every 1 000  — full statistics
MODE = "test"

# MPI processes per job
NPROCS = 1

# Thread limits — avoids oversubscription when running multiple jobs
ENV_EXTRA = {
    "OMP_NUM_THREADS":        "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
}

# Equilibration steps (NVT before MSD recording begins)
# 20 000 steps × 0.001 ps = 20 ps — enough to reach thermal equilibrium
EQUIL_STEPS = 20_000

# =============================================================================
#  SLAB GEOMETRY
# =============================================================================

def compute_slab_volume(datafile: str,
                        slab_atom_type: int = 1,
                        padding_top: float = 0.0) -> tuple[float, float, float]:
    """
    Parse a LAMMPS data file (atom_style atomic) and compute slab volume.

    The "slab volume" is defined as:
        area_xy × (z_max(Si) − z_min(Si) + padding_top)

    This is used only for the diagnostic slab pressure term and has no
    effect on the MSD or diffusion coefficient.

    Parameters
    ----------
    datafile       : path to LAMMPS .lmp data file
    slab_atom_type : integer atom type used to define the slab (1 = Si)
    padding_top    : extra Å added to the top of the slab thickness

    Returns
    -------
    area [Å²], slab_thickness [Å], slab_volume [Å³]
    """
    xlo = xhi = ylo = yhi = None
    zs = []
    start = None

    with open(datafile, "r") as fh:
        lines = fh.readlines()

    # ---- parse box bounds and locate "Atoms" section ------------------------
    for idx, line in enumerate(lines[:200]):
        stripped = line.strip()
        if stripped.endswith("xlo xhi"):
            parts = stripped.split()
            xlo, xhi = float(parts[0]), float(parts[1])
        elif stripped.endswith("ylo yhi"):
            parts = stripped.split()
            ylo, yhi = float(parts[0]), float(parts[1])
        elif stripped.startswith("Atoms"):
            # LAMMPS format: "Atoms" line, one blank line, then data
            start = idx + 2
            break

    if start is None:
        raise RuntimeError(f"'Atoms' section not found in {datafile}")
    if None in (xlo, xhi, ylo, yhi):
        raise RuntimeError(f"Box bounds not fully parsed from {datafile}")

    # ---- parse atom lines ---------------------------------------------------
    # atom_style atomic format: id  type  x  y  z
    # Columns are positional; image flags (ix iy iz) may follow z.
    # We use parts[4] (0-indexed) for z — reliable regardless of image flags.
    for raw in lines[start:]:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 5:
            break           # end of Atoms block
        try:
            atype = int(parts[1])
            z     = float(parts[4])   # column 4 is always z in atomic style
        except (ValueError, IndexError):
            continue
        if atype == slab_atom_type:
            zs.append(z)

    if not zs:
        raise RuntimeError(
            f"No atoms of type {slab_atom_type} found in {datafile}"
        )

    area            = (xhi - xlo) * (yhi - ylo)
    slab_thickness  = (max(zs) - min(zs)) + padding_top
    slab_vol        = area * slab_thickness
    return area, slab_thickness, slab_vol


# =============================================================================
#  INPUT FILE GENERATION
# =============================================================================

def write_input(run_dir:       str,
                T:             float,
                timestep:      float,
                steps:         int,
                equil_steps:   int,
                thermo_every:  int,
                msd_every:     int,
                slab_vol:      float,
                grace_path:    str) -> None:
    """
    Write the LAMMPS input file  in.diffusion  inside run_dir.

    Corrections vs. original script
    --------------------------------
    * velocity all create  is placed AFTER minimize  (minimizer zeros velocities)
    * Equilibration phase (NVT) added before production MSD recording
    * compute stress/atom already includes kinetic contribution — removed the
      erroneous separate subtraction of kinetic energy in the pressure variable
    * compute msd com no  (com yes on a 1-atom group gives MSD = 0 always)
    * Unwrapped coordinates (xu yu zu) added to dump for independent checks
    * MSD file has a header line so the analyzer can parse columns reliably
    """
    # Nosé–Hoover damping time: 100 × dt  (recommended 0.1–1 ps)
    nhv_damp = 100.0 * timestep

    input_text = f"""# Auto-generated by run_diffusion.py  —  T = {T} K
# See in.diffusion.lammps for the fully annotated reference version.

units        metal
atom_style   atomic
boundary     p p p

read_data    {os.path.basename(STRUCT_FILE)}

pair_style   grace
pair_coeff   * * {grace_path} Si H

group  Si_atoms  type 1
group  H_atom    type 2

# ---- Stage 1: energy minimization ------------------------------------------
min_style    cg
minimize     1.0e-4  1.0e-6  10000  100000

reset_timestep  0
timestep        {timestep}

# velocity AFTER minimization: minimizer resets all velocities to zero
velocity  all  create  {T}  12345  mom yes  rot yes  dist gaussian

# ---- Stage 2: NVT equilibration  ({equil_steps} steps = {equil_steps*timestep:.1f} ps) ----
fix  eq  all  nvt  temp  {T}  {T}  {nhv_damp}

thermo          500
thermo_style    custom  step  temp  etotal  press
thermo_modify   flush yes

run  {equil_steps}

unfix  eq
reset_timestep  0    # MSD t=0 starts here

# ---- Stage 3: production NVT  ({steps} steps = {steps*timestep:.1f} ps) ----------

# MSD of H atom
# com no: mandatory for a single-atom group —
#         com yes would subtract the atom's own displacement → MSD always 0
compute  msd_h  H_atom  msd  com no

variable  msd_tot  equal  c_msd_h[4]
variable  msd_x    equal  c_msd_h[1]
variable  msd_y    equal  c_msd_h[2]
variable  msd_z    equal  c_msd_h[3]
variable  s        equal  step

# slab pressure (diagnostic; stress/atom already includes kinetic term)
variable  SLABVOL  equal  {slab_vol:.8e}
compute  stress_pa   all  stress/atom  NULL
compute  virial_sum  all  reduce  sum  c_stress_pa[1]  c_stress_pa[2]  c_stress_pa[3]
variable  press_slab  equal  -(c_virial_sum[1]+c_virial_sum[2]+c_virial_sum[3]) &
                              /(3.0*v_SLABVOL)

thermo          {thermo_every}
thermo_style    custom  step  temp  etotal  v_press_slab  &
                v_msd_tot  v_msd_x  v_msd_y  v_msd_z
thermo_modify   flush yes

# trajectory: x y z = wrapped,  xu yu zu = unwrapped (PBC-corrected)
dump  traj  all  custom  {thermo_every}  dump.atom &
      id  type  x  y  z  xu  yu  zu
dump_modify  traj  flush yes

# MSD file — columns: step  msd_total  msd_x  msd_y  msd_z  [all in Å²]
fix  msd_out  all  print  {msd_every} &
     "${{s}} ${{msd_tot}} ${{msd_x}} ${{msd_y}} ${{msd_z}}" &
     file  msd_{T}K.dat  screen no &
     title "# step  msd_total[A^2]  msd_x[A^2]  msd_y[A^2]  msd_z[A^2]"

fix  nvt_prod  all  nvt  temp  {T}  {T}  {nhv_damp}

run  {steps}

write_data  final_state_{T}K.lmp
"""
    with open(os.path.join(run_dir, "in.diffusion"), "w") as fh:
        fh.write(input_text)


# =============================================================================
#  JOB LAUNCHER
# =============================================================================

def launch_nohup(run_dir: str, nprocs: int) -> None:
    """
    Launch LAMMPS in the background using nohup + mpirun.
    stdout and stderr are redirected to  log.lammps.
    """
    if not os.path.isfile(LAMMPS_EXE):
        raise FileNotFoundError(
            f"LAMMPS executable not found: {LAMMPS_EXE}\n"
            "Update LAMMPS_EXE at the top of this script."
        )

    cmd = (
        f"nohup mpirun -np {nprocs} {shlex.quote(LAMMPS_EXE)} "
        f"-in in.diffusion > log.lammps 2>&1 &"
    )
    env = os.environ.copy()
    env.update(ENV_EXTRA)
    subprocess.Popen(cmd, cwd=run_dir, shell=True, env=env)


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    # ---- slab geometry -------------------------------------------------------
    area, slab_thick, slab_vol = compute_slab_volume(
        STRUCT_FILE, slab_atom_type=1, padding_top=0.0
    )
    print(
        f"Slab geometry from {STRUCT_FILE}:\n"
        f"  area      = {area:.4f} Å²\n"
        f"  thickness = {slab_thick:.4f} Å  (Si z_max − z_min)\n"
        f"  volume    = {slab_vol:.4e} Å³\n"
    )

    # ---- run parameters ------------------------------------------------------
    if MODE == "test":
        def steps_for(T: float) -> int:
            return 2_000
        thermo_every = 100
        msd_every    = 100
        print("Mode: TEST  (2 000 steps per temperature)\n")
    else:
        def steps_for(T: float) -> int:
            # shorter timestep at high T → more steps needed for same physical time
            return 7_000_000 if T <= 1000 else 14_000_000
        thermo_every = 1_000
        msd_every    = 1_000
        print("Mode: PRODUCTION\n")

    # ---- launch one job per temperature --------------------------------------
    os.makedirs(RUN_FOLDER, exist_ok=True)

    for T in TEMPERATURES:
        run_dir = os.path.join(RUN_FOLDER, STRUCT_NAME, f"T{T}K")
        os.makedirs(run_dir, exist_ok=True)

        # shorter timestep at high T avoids integrator instability
        timestep = 0.001 if T <= 1000 else 0.0005
        steps    = steps_for(T)

        write_input(
            run_dir      = run_dir,
            T            = T,
            timestep     = timestep,
            steps        = steps,
            equil_steps  = EQUIL_STEPS,
            thermo_every = thermo_every,
            msd_every    = msd_every,
            slab_vol     = slab_vol,
            grace_path   = GRACE_PATH,
        )
        subprocess.run(["cp", STRUCT_FILE, run_dir], check=True)

        print(f"  Launching T = {T:5d} K  |  {steps:>10,} steps  |  {run_dir}")
        launch_nohup(run_dir, NPROCS)

    print(
        f"\nAll {len(TEMPERATURES)} jobs launched in background (nohup).\n"
        f"\nMonitor progress:\n"
        f"  tail -f {RUN_FOLDER}/{STRUCT_NAME}/T700K/log.lammps\n"
        f"\nCheck all logs:\n"
        f"  for d in {RUN_FOLDER}/{STRUCT_NAME}/T*/; do"
        f" echo \"=== $d ===\"; tail -3 \"$d/log.lammps\"; done\n"
        f"\nWhen done, run:\n"
        f"  python analyze_msd.py"
    )


if __name__ == "__main__":
    main()
