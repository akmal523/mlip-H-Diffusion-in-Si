import os
import subprocess
import shlex

struct_file = "si_with_h.lmp"          # LAMMPS data file 
struct_name = "Si64H1_box"             # label for folders
temperatures = [700, 800, 1000, 1200, 1500]
run_folder = "diffusion_runs"

# "test" = quick verification (short runs, frequent output)
# "production" = long diffusion runs
mode = "test"   # "test" or "production"

# MPI ranks
nprocs = 1

# TensorFlow/OpenMP threads 
env_extra = {
    "OMP_NUM_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
}

os.makedirs(run_folder, exist_ok=True)

def compute_slab_volume(datafile, slab_atom_type=1, padding_top=0.0):
    """
    Read LAMMPS data file and compute slab volume = area_xy * slab_thickness.
    slab_atom_type: integer for atoms used to define slab z extent (Si)
    padding_top: optional extra Angstroms to add on top (e.g., H spacing)
    Returns: area (Ang^2), slab_thickness (Ang), slab_vol (Ang^3)
    """
    xlo = xhi = ylo = yhi = zlo = zhi = None
    zs = []
    start = None

    with open(datafile, 'r') as f:
        lines = f.readlines()

    # Parse box bounds and find 'Atoms' section index
    for idx, line in enumerate(lines[:200]):
        l = line.strip()
        if l.endswith('xlo xhi'):
            parts = l.split()
            xlo, xhi = float(parts[0]), float(parts[1])
        if l.endswith('ylo yhi'):
            parts = l.split()
            ylo, yhi = float(parts[0]), float(parts[1])
        if l.endswith('zlo zhi'):
            parts = l.split()
            zlo, zhi = float(parts[0]), float(parts[1])
        if l == 'Atoms' or l.startswith('Atoms'):
            # atoms block follows;
            # it finds next non-empty line after this and treat as header skip
            start = idx + 2
            break

    if start is None:
        raise RuntimeError("Could not find 'Atoms' section in data file.")

    # Parse atoms lines robustly: support "atomic" (id type x y z) and "full"/"molecular" variants that might include extra columns.
    for L in lines[start:]:
        if not L.strip():
            continue
        parts = L.split()
        # attempt to find type and z coordinate: assume type is second token
        try:
            atype = int(parts[1])
        except Exception:
            continue
        # try to read last 3 as x y z
        try:
            z = float(parts[-1])
        except Exception:
            # fallback: use 5th column
            try:
                z = float(parts[4])
            except Exception:
                continue
        if atype == slab_atom_type:
            zs.append(z)

    if not (xlo is not None and xhi is not None and ylo is not None and yhi is not None and zs):
        raise RuntimeError("Could not parse box bounds or slab atom z coordinates from data file.")

    area = (xhi - xlo) * (yhi - ylo)   # Ang^2
    zmin, zmax = min(zs), max(zs)
    slab_thickness = (zmax - zmin) + padding_top   # Å
    slab_vol = area * slab_thickness               # Å^3

    return area, slab_thickness, slab_vol


def write_input(run_dir, T, timestep, steps, thermo_every, msd_every, slab_vol):
    """Create the LAMMPS input file in run_dir for temperature T."""
    # Minimization iterations - longer to get better relaxation
    min_iter = 10000
    min_eval = 100000

    in_text = f"""units           metal
atom_style      atomic
read_data       {os.path.basename(struct_file)}

# slab volume computed by Python (Ang^3)
variable SLABVOL equal {slab_vol:.8e}

# GRACE potential - map: Si H (type 1 = Si, type 2 = H)
pair_style      grace
pair_coeff      * * /home/akmal.razikulov/.cache/grace/GRACE-1L-OAM Si H

# H is atom type 2 in your data file
group           Hatom type 2

timestep        {timestep}
variable        T equal {T}

# initial velocities
velocity        all create ${{T}} 12345 mom yes rot yes dist gaussian

# relaxation 
min_style       cg
minimize        1.0e-4 1.0e-6 {min_iter} {min_eval}

# prepare for MD
# compute kinetic terms early
compute         ke all ke/atom
compute         tot_ekin all reduce sum c_ke

# per-atom virial (returns components: xx yy zz xy xz yz)
compute         peratom all stress/atom NULL

# sum virial components over all atoms
compute         ssum all reduce sum c_peratom[1] c_peratom[2] c_peratom[3]

# virial-based slab pressure
variable        slab_virial equal -(c_ssum[1] + c_ssum[2] + c_ssum[3])/(3.0 * v_SLABVOL)

# kinetic contribution (total kinetic energy)
variable        kin_energy equal c_tot_ekin

# total slab pressure
variable        slab_press_total equal v_slab_virial - (v_kin_energy)/(3.0 * v_SLABVOL)

# MSD of hydrogen atoms relative to COM
compute         msd_h Hatom msd com yes

# convenience variables
variable        msd equal c_msd_h[4]
variable        s    equal step

# output control
thermo          {thermo_every}
thermo_style    custom step temp etotal v_slab_virial v_slab_press_total v_msd
thermo_modify   flush yes

# trajectory (reduce size during production runs as needed)
dump            1 all custom {thermo_every} dump.atom id type x y z

# write MSD (step, msd) frequently
fix             msd_out all print {msd_every} "${{s}} ${{msd}}" file msd_h_{T}K.dat screen no

# NVT dynamics
fix             1 all nvt temp ${{T}} ${{T}} 0.1

# run
run             {steps}
"""
    with open(os.path.join(run_dir, "in.diffusion"), "w") as f:
        f.write(in_text)


def launch_nohup(run_dir, nprocs):
    """Launch LAMMPS via nohup mpirun in background, redirect to log.lammps."""
    lmp_exe = os.path.expanduser("~/MarquesN/lammps/build/lmp")
    # ensure executable exists
    if not os.path.isfile(lmp_exe):
        raise RuntimeError(f"LAMMPS executable not found at: {lmp_exe}")
    cmd = f"nohup mpirun -np {nprocs} {shlex.quote(lmp_exe)} -in in.diffusion > log.lammps 2>&1 &"
    env = os.environ.copy()
    env.update(env_extra)
    subprocess.Popen(cmd, cwd=run_dir, shell=True, env=env)


def main():
    # compute slab geometry from the provided data file
    area, slab_thick, slab_vol = compute_slab_volume(struct_file, slab_atom_type=1, padding_top=0.0)
    print(f"Detected slab area = {area:.6f} Å^2, thickness = {slab_thick:.6f} Å, volume = {slab_vol:.6e} Å^3")

    # choose run sizes
    if mode == "test":
        steps_by_T = lambda T: 2000
        thermo_every = 100
        msd_every = 100
    else:
        steps_by_T = lambda T: (7_000_000 if T <= 1000 else 14_000_000)
        thermo_every = 1000
        msd_every = 1000

    for T in temperatures:
        run_dir = os.path.join(run_folder, struct_name, f"T{T}K")
        os.makedirs(run_dir, exist_ok=True)

        timestep = 0.001 if T <= 1000 else 0.0005
        steps = steps_by_T(T)

        # write input and copy structure
        write_input(run_dir, T, timestep, steps, thermo_every, msd_every, slab_vol)
        subprocess.run(["cp", struct_file, run_dir], check=True)

        print(f"Launching diffusion for {struct_name} at {T} K -> {run_dir}")
        launch_nohup(run_dir, nprocs)

    print("\n All jobs launched in background with nohup.")
    print("tail -f diffusion_runs/{}/T700K/log.lammps".format(struct_name))


if __name__ == "__main__":
    main()
