"""
create_structures.py
====================
Generates two LAMMPS structure files for the H-diffusion study:

  si_with_h_surface.lmp       — H adsorbed on top Si surface
                                  (current literature variant; H on exterior)
  si_with_h_interstitial.lmp  — H at a tetrahedral interstitial site
                                  inside the Si bulk
                                  (bulk diffusion variant; recommended by
                                   professor for studying true bulk migration)

Physical background
-------------------
Surface adsorption (current variant):
  H sits above the outermost Si layer.  The simulation box has a vacuum
  gap in z to prevent H from interacting with periodic images.
  Relevant for surface/near-surface H trapping and desorption.

Tetrahedral interstitial (bulk variant):
  H occupies a tetrahedral void (T-site) inside the Si lattice — the
  largest interstitial cavity in the diamond cubic structure.
  The T-site has 4 Si neighbours at equal distance (~2.35 Å before
  relaxation, ~2.0 Å after).  The box is fully periodic (no vacuum gap)
  because H is entirely surrounded by bulk Si.
  Relevant for bulk diffusion, migration barriers, and trapping at
  defects — the physically correct setup for studying H transport in Si.

Both files use the same 64-Si 2×2×2 diamond cubic supercell.
LAMMPS will further relax both structures via energy minimization at the
start of each diffusion run (in.diffusion.lammps stage 1).

Usage
-----
    python create_structures.py

Output
------
  si_with_h_surface.lmp          (copy/rename of original si_with_h.lmp)
  si_with_h_interstitial.lmp     (new file with H at T-site)
"""

import os
import shutil
import numpy as np

# =============================================================================
#  CONFIGURATION
# =============================================================================

BASE_STRUCT   = "si_with_h.lmp"       # original structure (H on surface)
OUT_SURFACE   = "si_with_h_surface.lmp"
OUT_INTERSTITIAL = "si_with_h_interstitial.lmp"

# =============================================================================
#  PARSER
# =============================================================================

def parse_lammps_data(path: str) -> dict:
    """
    Parse a LAMMPS data file (atom_style atomic).
    Returns dict with keys: header_lines, box, masses, atoms
      atoms: list of (id, type, x, y, z)
    """
    with open(path) as fh:
        lines = fh.readlines()

    box    = {}
    masses = []
    atoms  = []
    title  = lines[0].strip()
    start  = None

    for i, line in enumerate(lines):
        l = line.strip()
        if l.endswith("xlo xhi"):
            p = l.split(); box["x"] = (float(p[0]), float(p[1]))
        elif l.endswith("ylo yhi"):
            p = l.split(); box["y"] = (float(p[0]), float(p[1]))
        elif l.endswith("zlo zhi"):
            p = l.split(); box["z"] = (float(p[0]), float(p[1]))
        elif l.startswith("Masses"):
            # read until blank line
            j = i + 2
            while j < len(lines) and lines[j].strip():
                masses.append(lines[j].strip())
                j += 1
        elif l.startswith("Atoms"):
            start = i + 2
            break

    for line in lines[start:]:
        p = line.split()
        if not p or p[0].startswith("#"):
            continue
        if len(p) < 5:
            break
        try:
            atoms.append((int(p[0]), int(p[1]),
                          float(p[2]), float(p[3]), float(p[4])))
        except ValueError:
            break

    return {"title": title, "box": box, "masses": masses, "atoms": atoms}


def write_lammps_data(path: str, data: dict, title: str) -> None:
    """Write a LAMMPS data file (atom_style atomic)."""
    atoms  = data["atoms"]
    box    = data["box"]
    masses = data["masses"]

    n_atoms = len(atoms)
    n_types = max(t for _, t, *_ in atoms)

    with open(path, "w") as fh:
        fh.write(f"{title}\n\n")
        fh.write(f"{n_atoms} atoms\n")
        fh.write(f"{n_types} atom types\n\n")
        fh.write(f"{box['x'][0]:.6f} {box['x'][1]:.6f} xlo xhi\n")
        fh.write(f"{box['y'][0]:.6f} {box['y'][1]:.6f} ylo yhi\n")
        fh.write(f"{box['z'][0]:.6f} {box['z'][1]:.6f} zlo zhi\n")
        fh.write("\nMasses\n\n")
        for m in masses:
            fh.write(f"{m}\n")
        fh.write("\nAtoms\n\n")
        for aid, atype, x, y, z in atoms:
            fh.write(f"{aid:6d}  {atype}  {x:.6f}  {y:.6f}  {z:.6f}\n")


# =============================================================================
#  FIND BEST TETRAHEDRAL INTERSTITIAL SITE
# =============================================================================

def find_t_site(si_positions: np.ndarray,
                box:          dict) -> np.ndarray:
    """
    Return the tetrahedral interstitial (T-site) coordinates that are
    maximally distant from all Si atoms, with PBC in x and y.

    In the diamond cubic structure the T-site lies at the centre of a
    regular tetrahedron formed by 4 Si atoms.  For a 2×2×2 supercell
    with a = 5.431 Å the T-sites are located at multiples of a/2
    offset by a/4 in each direction.  We enumerate all such candidates
    inside the Si slab and pick the one with the largest nearest-Si
    distance.
    """
    a  = 5.431           # Si lattice constant [Å]
    Lx = box["x"][1] - box["x"][0]
    Ly = box["y"][1] - box["y"][0]

    z_lo = box["z"][0]
    # Use only the Si-occupied z range (not the vacuum)
    z_hi_si = float(np.max(si_positions[:, 2]))

    candidates = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                x = box["x"][0] + (i + 0.5) * a / 2
                y = box["y"][0] + (j + 0.5) * a / 2
                z = box["z"][0] + (k + 0.5) * a / 2
                # Must be strictly inside the Si bulk
                if z_lo + 0.5 < z < z_hi_si - 0.5:
                    candidates.append(np.array([x, y, z]))

    if not candidates:
        raise RuntimeError("No T-site candidates found inside Si slab.")

    def min_dist(pt: np.ndarray) -> float:
        dx = si_positions[:, 0] - pt[0]
        dy = si_positions[:, 1] - pt[1]
        dz = si_positions[:, 2] - pt[2]
        dx -= Lx * np.round(dx / Lx)
        dy -= Ly * np.round(dy / Ly)
        return float(np.min(np.sqrt(dx**2 + dy**2 + dz**2)))

    best     = max(candidates, key=min_dist)
    best_d   = min_dist(best)
    print(f"  T-site found: ({best[0]:.4f}, {best[1]:.4f}, {best[2]:.4f}) Å")
    print(f"  Nearest Si at {best_d:.3f} Å  (relaxes to ~2.0 Å after minimization)")
    return best


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    if not os.path.isfile(BASE_STRUCT):
        raise FileNotFoundError(
            f"Base structure '{BASE_STRUCT}' not found.\n"
            "This file must be present in the working directory."
        )

    data = parse_lammps_data(BASE_STRUCT)

    si_atoms = [(aid, t, x, y, z) for aid, t, x, y, z in data["atoms"] if t == 1]
    si_pos   = np.array([(x, y, z) for _, _, x, y, z in si_atoms])

    # ------------------------------------------------------------------
    # Variant 1 — surface  (copy original, just rename for consistency)
    # ------------------------------------------------------------------
    print("Creating surface variant ...")
    shutil.copy(BASE_STRUCT, OUT_SURFACE)
    print(f"  Written: {OUT_SURFACE}")

    # ------------------------------------------------------------------
    # Variant 2 — tetrahedral interstitial
    # ------------------------------------------------------------------
    print("\nCreating interstitial variant ...")

    t_site = find_t_site(si_pos, data["box"])

    # Build atom list: all Si atoms + 1 H at T-site
    # New atom id = max existing id + 1
    new_id = max(aid for aid, *_ in data["atoms"]) + 1
    new_atoms = list(si_atoms) + [(new_id, 2, t_site[0], t_site[1], t_site[2])]

    # Interstitial box: fully periodic — no vacuum gap in z.
    # Shrink z to match x and y (Si lattice only, no surface extension).
    box_int = {
        "x": data["box"]["x"],
        "y": data["box"]["y"],
        "z": (data["box"]["z"][0],
              data["box"]["z"][0] + (data["box"]["x"][1] - data["box"]["x"][0])),
    }

    data_int = {
        "atoms":  new_atoms,
        "box":    box_int,
        "masses": data["masses"],
    }

    write_lammps_data(
        OUT_INTERSTITIAL,
        data_int,
        title=(
            "LAMMPS data file: 64 Si + 1 H at tetrahedral interstitial (T-site) — "
            "bulk diffusion variant"
        ),
    )
    print(f"  Written: {OUT_INTERSTITIAL}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("Structure files ready")
    print("=" * 55)
    print(f"  {OUT_SURFACE}")
    print(f"    H position: on top Si surface (z ≈ 12.43 Å, above slab)")
    print(f"    Box z:      {data['box']['z'][0]:.3f} → {data['box']['z'][1]:.3f} Å  (vacuum gap)")
    print(f"    Boundary:   periodic x,y,z — vacuum gap prevents image interaction")
    print()
    print(f"  {OUT_INTERSTITIAL}")
    print(f"    H position: T-site inside Si bulk")
    print(f"    Box z:      {box_int['z'][0]:.3f} → {box_int['z'][1]:.3f} Å  (no vacuum)")
    print(f"    Boundary:   fully periodic x,y,z — bulk Si")
    print()
    print("Next step:")
    print("  Set STRUCT_VARIANT in run_diffusion.py and launch simulations.")
    print('    STRUCT_VARIANT = "surface"       → uses si_with_h_surface.lmp')
    print('    STRUCT_VARIANT = "interstitial"  → uses si_with_h_interstitial.lmp')
    print('    STRUCT_VARIANT = "both"          → runs both in separate folders')


if __name__ == "__main__":
    main()
