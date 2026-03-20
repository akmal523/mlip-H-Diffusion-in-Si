"""
check_environment.py
====================
Verifies that the LAMMPS executable and the GRACE-1L-OAM potential
directory are accessible on this machine.

  * If found automatically → confirms and saves paths.
  * If not found           → asks the user to enter paths manually.
  * Saves result to        → env_config.json (read by run_diffusion.py).

Usage
-----
    python check_environment.py

Run this once before running run_diffusion.py on a new machine.
"""

import os
import json
import shutil
import subprocess
import sys

CONFIG_FILE = "env_config.json"

# Candidate executable names to search in PATH
LAMMPS_CANDIDATES = ["lmp", "lmp_mpi", "lmp_serial", "lammps"]

# Sub-paths inside a GRACE directory that must exist for the model to be valid
GRACE_REQUIRED_FILES = ["Si.yaml", "H.yaml"]   # adjust if your model uses different names


# =============================================================================
#  HELPERS
# =============================================================================

def _bold(text: str) -> str:
    """ANSI bold — works on Linux/macOS terminals."""
    return f"\033[1m{text}\033[0m"


def _green(text: str) -> str:
    return f"\033[32m{text}\033[0m"


def _yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m"


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m"


def find_lammps_in_path() -> str | None:
    """Return the first LAMMPS executable found in PATH, or None."""
    for name in LAMMPS_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None


def find_lammps_common_locations() -> str | None:
    """Check a list of common build locations."""
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "lammps", "build", "lmp"),
        os.path.join(home, "lammps", "build", "lmp_mpi"),
        os.path.join(home, "MarquesN", "lammps", "build", "lmp"),
        os.path.join(home, "MarquesN", "lammps", "build", "lmp_mpi"),
        "/usr/local/bin/lmp",
        "/usr/bin/lmp",
        "/opt/lammps/bin/lmp",
    ]
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def verify_lammps_exe(path: str) -> tuple[bool, str]:
    """
    Try to run  lmp -help  and confirm LAMMPS responds.
    Returns (ok: bool, version_string: str).
    """
    try:
        result = subprocess.run(
            [path, "-help"],
            capture_output=True, text=True, timeout=10,
        )
        # LAMMPS prints version to stdout or stderr
        combined = result.stdout + result.stderr
        for line in combined.splitlines():
            if "LAMMPS" in line and ("version" in line.lower() or "(" in line):
                return True, line.strip()
        # If it ran without error but we couldn't parse version, still accept it
        if result.returncode in (0, 1):
            return True, "(version string not parsed)"
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        pass
    return False, ""


def find_grace_in_common_locations() -> str | None:
    """Check common GRACE cache and build locations."""
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".cache", "grace", "GRACE-1L-OAM"),
        os.path.join(home, ".grace", "GRACE-1L-OAM"),
        os.path.join(home, "GRACE-1L-OAM"),
        os.path.join(home, "MarquesN", "GRACE-1L-OAM"),
        "/opt/grace/GRACE-1L-OAM",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def verify_grace_path(path: str) -> tuple[bool, list[str]]:
    """
    Check that the directory exists and contains expected element files.
    Returns (ok: bool, missing_files: list[str]).
    """
    if not os.path.isdir(path):
        return False, ["(directory does not exist)"]
    missing = [
        f for f in GRACE_REQUIRED_FILES
        if not os.path.isfile(os.path.join(path, f))
    ]
    # Warn but don't block if files are missing — naming conventions vary
    return len(missing) == 0, missing


def prompt_path(label: str, hint: str = "") -> str:
    """Prompt the user to enter a path, with an optional hint."""
    if hint:
        print(f"  Hint: {_yellow(hint)}")
    while True:
        raw = input(f"  Enter {label}: ").strip()
        if raw:
            expanded = os.path.expanduser(raw)
            return expanded
        print("  Path cannot be empty. Please try again.")


# =============================================================================
#  MAIN CHECKS
# =============================================================================

def check_lammps() -> str:
    """
    Locate the LAMMPS executable.
    Returns the confirmed path.
    """
    print(_bold("\n[1/2] Checking LAMMPS executable"))

    # Auto-detect
    found = find_lammps_in_path() or find_lammps_common_locations()
    if found:
        ok, version = verify_lammps_exe(found)
        if ok:
            print(f"  {_green('FOUND')}  {found}")
            print(f"  {version}")
            return found
        else:
            print(f"  Located at {found} but could not execute it.")

    print(f"  {_yellow('NOT FOUND')}  (searched PATH + common locations)")
    print(
        "\n  LAMMPS must be compiled from source with the GRACE pair style.\n"
        "  Build instructions: https://docs.lammps.org/Build.html\n"
        "  GRACE pair style:   https://github.com/ICAMS/grace-tensorpotential\n"
    )

    while True:
        exe = prompt_path(
            "full path to LAMMPS executable",
            hint="e.g.  ~/lammps/build/lmp"
        )
        ok, version = verify_lammps_exe(exe)
        if ok:
            print(f"  {_green('OK')}  {version}")
            return exe
        else:
            print(
                f"  {_red('Could not run')} {exe}\n"
                "  Make sure the file exists and is executable (chmod +x)."
            )
            retry = input("  Try a different path? [Y/n]: ").strip().lower()
            if retry == "n":
                print("  Saving path anyway — verify manually before running simulations.")
                return exe


def check_grace() -> str:
    """
    Locate the GRACE-1L-OAM potential directory.
    Returns the confirmed path.
    """
    print(_bold("\n[2/2] Checking GRACE-1L-OAM potential"))

    # Auto-detect
    found = find_grace_in_common_locations()
    if found:
        ok, missing = verify_grace_path(found)
        if ok:
            print(f"  {_green('FOUND')}  {found}")
            return found
        else:
            print(
                f"  Found directory {found} but missing expected files: "
                f"{', '.join(missing)}"
            )
            print("  This may be fine if your GRACE model uses different filenames.")
            accept = input("  Use this path anyway? [Y/n]: ").strip().lower()
            if accept != "n":
                return found

    print(f"  {_yellow('NOT FOUND')}  (searched common cache locations)")
    print(
        "\n  Download GRACE-1L-OAM from:\n"
        "  https://github.com/ICAMS/grace-tensorpotential/releases\n"
        "  or your cluster's model repository.\n"
    )

    while True:
        path = prompt_path(
            "path to GRACE-1L-OAM directory",
            hint="e.g.  ~/.cache/grace/GRACE-1L-OAM"
        )
        ok, missing = verify_grace_path(path)
        if ok:
            print(f"  {_green('OK')}  Directory confirmed.")
            return path
        else:
            print(
                f"  {_yellow('Warning')}  Directory not found or missing files: "
                f"{', '.join(missing)}"
            )
            accept = input("  Save this path anyway? [Y/n]: ").strip().lower()
            if accept != "n":
                return path


# =============================================================================
#  SAVE / LOAD CONFIG
# =============================================================================

def save_config(lammps_exe: str, grace_path: str) -> None:
    config = {
        "lammps_exe":  lammps_exe,
        "grace_path":  grace_path,
    }
    with open(CONFIG_FILE, "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"\n  Configuration saved to  {_bold(CONFIG_FILE)}")
    print(f"    lammps_exe : {lammps_exe}")
    print(f"    grace_path : {grace_path}")
    print(
        "\n  run_diffusion.py will read this file automatically.\n"
        "  Re-run check_environment.py any time to update the paths."
    )


def load_config() -> dict | None:
    """
    Load saved config.  Returns dict or None if file doesn't exist.
    Called by run_diffusion.py.
    """
    if not os.path.isfile(CONFIG_FILE):
        return None
    with open(CONFIG_FILE, "r") as fh:
        return json.load(fh)


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main() -> None:
    print("=" * 55)
    print(_bold("  H-diffusion-in-Si  —  Environment Check"))
    print("=" * 55)

    # If config already exists, show it and ask whether to re-check
    existing = load_config()
    if existing:
        print(f"\n  Existing config found ({CONFIG_FILE}):")
        print(f"    lammps_exe : {existing.get('lammps_exe', 'N/A')}")
        print(f"    grace_path : {existing.get('grace_path', 'N/A')}")
        recheck = input("\n  Re-run environment check? [y/N]: ").strip().lower()
        if recheck != "y":
            print("  Using existing config.  Done.")
            sys.exit(0)

    lammps_exe = check_lammps()
    grace_path = check_grace()
    save_config(lammps_exe, grace_path)

    print("\n" + "=" * 55)
    print(_green(_bold("  Environment check complete.")))
    print("=" * 55)
    print("\n  Next steps:")
    print("    1.  python run_diffusion.py   # launch simulations")
    print("    2.  python analyze_msd.py     # analyze results")
    print(
        "\n  No LAMMPS/GRACE? Run analyze_msd.py on the included\n"
        "  pre-simulated data in  presimulated/  — see README.\n"
    )


if __name__ == "__main__":
    main()
