#!/usr/bin/env bash
# =============================================================================
#  monitor_jobs.sh
#  Monitor and manage LAMMPS diffusion runs for H-diffusion-in-Si.
#
#  Usage:
#    ./monitor_jobs.sh            # show status of all temperature runs
#    ./monitor_jobs.sh --watch    # refresh every 30 s until all done
#    ./monitor_jobs.sh --errors   # print any LAMMPS errors/warnings
#    ./monitor_jobs.sh --clean    # remove incomplete run directories
# =============================================================================

set -euo pipefail

# --- configuration -----------------------------------------------------------
RUN_BASE="diffusion_runs/Si64H1_box"
TEMPERATURES=(700 800 1000 1200 1500)
WATCH_INTERVAL=30   # seconds between refreshes in --watch mode

# ANSI colours
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'   # reset

# =============================================================================
#  HELPERS
# =============================================================================

print_header() {
    echo ""
    echo -e "${BOLD}============================================================${NC}"
    echo -e "${BOLD}  H-diffusion-in-Si  —  Run Monitor${NC}"
    echo -e "${BOLD}  $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${BOLD}============================================================${NC}"
    echo ""
}

# Returns: "done" | "running" | "failed" | "missing"
run_status() {
    local dir="$1"
    local log="$dir/log.lammps"

    if [[ ! -d "$dir" ]]; then
        echo "missing"
        return
    fi

    if [[ ! -f "$log" ]]; then
        echo "missing"
        return
    fi

    # LAMMPS writes "Total wall time" at the very end on clean exit
    if grep -q "Total wall time" "$log" 2>/dev/null; then
        echo "done"
        return
    fi

    # Check for known error keywords
    if grep -qiE "^(ERROR|WARNING.*abort|Segmentation fault)" "$log" 2>/dev/null; then
        echo "failed"
        return
    fi

    # Log exists but no final line → still running (or killed mid-run)
    if pgrep -f "in.diffusion" >/dev/null 2>&1; then
        echo "running"
    else
        # Process not found → likely killed or not yet started
        echo "running"
    fi
}

# Extract the last thermo line: step and temperature
last_thermo() {
    local log="$1"
    # Thermo lines start with an integer step number
    grep -E "^[[:space:]]*[0-9]+" "$log" 2>/dev/null | tail -1 || true
}

# Count MSD data points written so far
msd_points() {
    local dir="$1"
    local T="$2"
    local dat="$dir/msd_${T}K.dat"
    if [[ -f "$dat" ]]; then
        grep -cv "^#" "$dat" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

# Estimate progress % from step count in log
progress_pct() {
    local log="$1"
    local total_steps="$2"
    local last_step
    last_step=$(grep -E "^[[:space:]]*[0-9]+" "$log" 2>/dev/null \
                | awk '{print $1}' | tail -1)
    if [[ -z "$last_step" || "$total_steps" -eq 0 ]]; then
        echo "0"
        return
    fi
    echo $(( last_step * 100 / total_steps ))
}

# Production step counts (must match run_diffusion.py)
expected_steps() {
    local T="$1"
    if (( T <= 1000 )); then echo 7000000; else echo 14000000; fi
}

# =============================================================================
#  STATUS TABLE
# =============================================================================

show_status() {
    print_header

    local all_done=true

    printf "  %-10s  %-10s  %-8s  %-10s  %-s\n" \
           "Temp (K)" "Status" "Progress" "MSD pts" "Last log line"
    printf "  %-10s  %-10s  %-8s  %-10s  %-s\n" \
           "--------" "------" "--------" "-------" "-------------"

    for T in "${TEMPERATURES[@]}"; do
        local dir="${RUN_BASE}/T${T}K"
        local log="${dir}/log.lammps"
        local status
        status=$(run_status "$dir")

        local colour="$NC"
        local status_str="$status"
        case "$status" in
            done)    colour="$GREEN"  ;;
            running) colour="$CYAN"   ; all_done=false ;;
            failed)  colour="$RED"    ; all_done=false ;;
            missing) colour="$YELLOW" ; all_done=false ;;
        esac

        local pct="-"
        local pts="-"
        local last_line="-"

        if [[ -f "$log" ]]; then
            local steps
            steps=$(expected_steps "$T")
            pct="$(progress_pct "$log" "$steps")%"
            pts=$(msd_points "$dir" "$T")
            last_line=$(last_thermo "$log" | cut -c1-60)
        fi

        printf "  %-10s  ${colour}%-10s${NC}  %-8s  %-10s  %s\n" \
               "${T}" "$status_str" "$pct" "$pts" "$last_line"
    done

    echo ""

    if $all_done; then
        echo -e "  ${GREEN}${BOLD}All runs finished.${NC}"
        echo ""
        echo -e "  Run analysis:"
        echo -e "    ${BOLD}python analyze_msd.py${NC}"
    else
        echo -e "  ${CYAN}Tip:${NC} tail a specific log:"
        echo -e "    tail -f ${RUN_BASE}/T700K/log.lammps"
    fi
    echo ""
}

# =============================================================================
#  ERROR / WARNING REPORT
# =============================================================================

show_errors() {
    print_header
    echo -e "  ${BOLD}Errors and warnings in LAMMPS logs${NC}"
    echo ""

    local found=false

    for T in "${TEMPERATURES[@]}"; do
        local log="${RUN_BASE}/T${T}K/log.lammps"
        if [[ ! -f "$log" ]]; then
            continue
        fi

        local hits
        hits=$(grep -niE "^(ERROR|WARNING)" "$log" 2>/dev/null || true)

        if [[ -n "$hits" ]]; then
            found=true
            echo -e "  ${YELLOW}T = ${T} K${NC}  (${log})"
            echo "$hits" | while IFS= read -r line; do
                echo "    $line"
            done
            echo ""
        fi
    done

    if ! $found; then
        echo -e "  ${GREEN}No errors or warnings found in any log.${NC}"
    fi
    echo ""
}

# =============================================================================
#  CLEAN INCOMPLETE RUNS
# =============================================================================

clean_incomplete() {
    print_header
    echo -e "  ${BOLD}Scanning for incomplete run directories...${NC}"
    echo ""

    local found=false

    for T in "${TEMPERATURES[@]}"; do
        local dir="${RUN_BASE}/T${T}K"
        local status
        status=$(run_status "$dir")

        if [[ "$status" == "missing" ]]; then
            continue
        fi

        if [[ "$status" != "done" ]]; then
            found=true
            echo -e "  ${YELLOW}${dir}${NC}  (status: ${status})"
            read -rp "  Delete this directory? [y/N]: " confirm
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                rm -rf "$dir"
                echo -e "  ${RED}Deleted.${NC}"
            else
                echo "  Skipped."
            fi
            echo ""
        fi
    done

    if ! $found; then
        echo -e "  ${GREEN}No incomplete directories found.${NC}"
    fi
    echo ""
}

# =============================================================================
#  WATCH MODE
# =============================================================================

watch_mode() {
    while true; do
        clear
        show_status

        # Check if all done
        local all_done=true
        for T in "${TEMPERATURES[@]}"; do
            local status
            status=$(run_status "${RUN_BASE}/T${T}K")
            if [[ "$status" != "done" ]]; then
                all_done=false
                break
            fi
        done

        if $all_done; then
            echo -e "  ${GREEN}${BOLD}All jobs complete. Exiting watch mode.${NC}"
            echo ""
            break
        fi

        echo -e "  Refreshing every ${WATCH_INTERVAL}s  —  Ctrl+C to exit"
        sleep "$WATCH_INTERVAL"
    done
}

# =============================================================================
#  ENTRY POINT
# =============================================================================

case "${1:-}" in
    --watch)   watch_mode    ;;
    --errors)  show_errors   ;;
    --clean)   clean_incomplete ;;
    --help|-h)
        echo ""
        echo "Usage: $0 [option]"
        echo ""
        echo "  (no option)   Show status table for all temperature runs"
        echo "  --watch       Refresh status every ${WATCH_INTERVAL}s until all runs finish"
        echo "  --errors      Print LAMMPS errors and warnings from all logs"
        echo "  --clean       Interactively delete incomplete run directories"
        echo "  --help        Show this message"
        echo ""
        ;;
    "")        show_status   ;;
    *)
        echo "Unknown option: $1"
        echo "Run  $0 --help  for usage."
        exit 1
        ;;
esac
