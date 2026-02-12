#!/bin/bash

################################################################################
# GPUMD + VASP QM/MM Hybrid MD via MDI
#
# This script orchestrates a QM/MM simulation where:
#  - GPUMD runs as MDI ENGINE (MD integrator)
#  - VASP MDI driver runs as DRIVER (QM calculator)
#
# Usage:
#   bash run_mdi_vasp_gpumd.sh [options]
#
# Options:
#   --gpumd-bin <path>        Path to GPUMD binary (default: ./gpumd)
#   --run-in <file>           GPUMD input file (default: run.in)
#   --vasp-cmd <cmd>          VASP execution command (default: vasp_std)
#   --poscar <file>           POSCAR_template (default: POSCAR_template)
#   --port <port>             MDI TCP port (default: 8021)
#   --steps <N>               Number of MD steps (default: 100)
#   --timeout <secs>          VASP timeout (default: 3600)
#   --no-cleanup              Keep intermediate files
#   --verbose                 Enable verbose logging
#
# Example:
#   bash run_mdi_vasp_gpumd.sh --steps 50 --vasp-cmd "mpirun -n 8 vasp_std"
#
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Configuration defaults
GPUMD_BIN="./gpumd"
RUN_IN="run.in"
VASP_CMD="vasp_std"
POSCAR_TEMPLATE="POSCAR_template"
PORT=8021
STEPS=100
TIMEOUT=3600
CLEANUP=true
VERBOSE=false

# Function to print colored messages
log_info()   { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success(){ echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()  { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpumd-bin)    GPUMD_BIN="$2"; shift 2;;
        --run-in)       RUN_IN="$2"; shift 2;;
        --vasp-cmd)     VASP_CMD="$2"; shift 2;;
        --poscar)       POSCAR_TEMPLATE="$2"; shift 2;;
        --port)         PORT="$2"; shift 2;;
        --steps)        STEPS="$2"; shift 2;;
        --timeout)      TIMEOUT="$2"; shift 2;;
        --no-cleanup)   CLEANUP=false; shift;;
        --verbose)      VERBOSE=true; shift;;
        *) log_error "Unknown option: $1"; exit 1;;
    esac
done

log_info "==========================================================="
log_info "GPUMD + VASP QM/MM Hybrid Simulation via MDI"
log_info "==========================================================="
log_info "GPUMD binary:        $GPUMD_BIN"
log_info "GPUMD input:         $RUN_IN"
log_info "VASP command:        $VASP_CMD"
log_info "POSCAR template:     $POSCAR_TEMPLATE"
log_info "MDI port:            $PORT"
log_info "MD steps:            $STEPS"
log_info "VASP timeout:        $TIMEOUT seconds"
log_info "==========================================================="

# Check prerequisites
log_info "Checking prerequisites..."

if [ ! -f "$GPUMD_BIN" ]; then
    log_error "GPUMD binary not found: $GPUMD_BIN"
    exit 1
fi
log_info "✓ Found GPUMD binary: $GPUMD_BIN"

if [ ! -f "$RUN_IN" ]; then
    log_error "GPUMD input file not found: $RUN_IN"
    exit 1
fi
log_info "✓ Found GPUMD input: $RUN_IN"

if [ ! -f "$POSCAR_TEMPLATE" ]; then
    log_error "POSCAR_template not found: $POSCAR_TEMPLATE"
    exit 1
fi
log_info "✓ Found POSCAR_template: $POSCAR_TEMPLATE"

# Check if vasp_mdi_driver.py exists in current directory or script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DRIVER_SCRIPT="vasp_mdi_driver.py"

if [ ! -f "$DRIVER_SCRIPT" ]; then
    if [ -f "$SCRIPT_DIR/$DRIVER_SCRIPT" ]; then
        DRIVER_SCRIPT="$SCRIPT_DIR/$DRIVER_SCRIPT"
    else
        log_error "vasp_mdi_driver.py not found"
        exit 1
    fi
fi
log_info "✓ Found driver script: $DRIVER_SCRIPT"

# Check Python and MDI
if ! python3 -c "import mdi" 2>/dev/null; then
    log_error "MDI Library for Python not found. Install with: pip install mdi"
    exit 1
fi
log_info "✓ MDI Library available"

log_info "==========================================================="
log_info "Starting simulation..."
log_info "==========================================================="

# Create temp directory for intermediate files
TMPDIR=".gpumd_vasp_tmp"
mkdir -p "$TMPDIR"

# Create log directory that persists
LOGDIR=".gpumd_logs"
mkdir -p "$LOGDIR"

# Cleanup function
cleanup() {
    if [ "$CLEANUP" = true ]; then
        log_info "Cleaning up temporary files (keeping logs in $LOGDIR)..."
        rm -rf "$TMPDIR"
    else
        log_warn "Keeping temporary directory: $TMPDIR"
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Copy logs to persistent directory function
save_logs() {
    if [ -f "$TMPDIR/gpumd.log" ]; then
        cp "$TMPDIR/gpumd.log" "$LOGDIR/gpumd_$(date +%s).log"
    fi
    if [ -f "$TMPDIR/vasp_driver.log" ]; then
        cp "$TMPDIR/vasp_driver.log" "$LOGDIR/vasp_driver_$(date +%s).log"
    fi
}

# Start GPUMD as ENGINE in background
log_info ""
log_info "Step 1: Starting GPUMD as MDI ENGINE..."
log_info "Command: $GPUMD_BIN --mdi '-role ENGINE -name gpumd -method TCP -hostname localhost -port $PORT' -in $RUN_IN"

"$GPUMD_BIN" --mdi "-role ENGINE -name gpumd -method TCP -hostname localhost -port $PORT" \
    -in "$RUN_IN" > "$TMPDIR/gpumd.log" 2>&1 &
GPUMD_PID=$!
log_info "GPUMD started with PID: $GPUMD_PID"

# Give GPUMD time to initialize and listen on port
log_info "Waiting for GPUMD to initialize..."
sleep 3

# Check if GPUMD is still running
if ! kill -0 $GPUMD_PID 2>/dev/null; then
    log_error "GPUMD failed to start. Check log:"
    cat "$TMPDIR/gpumd.log"
    exit 1
fi
log_success "GPUMD is running"

# Start Python VASP driver as DRIVER
log_info ""
log_info "Step 2: Starting Python VASP MDI driver as DRIVER..."
log_info "Command: python3 $DRIVER_SCRIPT --vasp-cmd '$VASP_CMD' --poscar-template $POSCAR_TEMPLATE --port $PORT --steps $STEPS --timeout $TIMEOUT"

DRIVER_LOG="$TMPDIR/vasp_driver.log"
if [ "$VERBOSE" = true ]; then
    python3 "$DRIVER_SCRIPT" \
        --vasp-cmd "$VASP_CMD" \
        --poscar-template "$POSCAR_TEMPLATE" \
        --port "$PORT" \
        --steps "$STEPS" \
        --timeout "$TIMEOUT" | tee "$DRIVER_LOG"
    DRIVER_EXIT=$?
else
    python3 "$DRIVER_SCRIPT" \
        --vasp-cmd "$VASP_CMD" \
        --poscar-template "$POSCAR_TEMPLATE" \
        --port "$PORT" \
        --steps "$STEPS" \
        --timeout "$TIMEOUT" > "$DRIVER_LOG" 2>&1
    DRIVER_EXIT=$?
fi

log_info ""
log_info "Step 3: Waiting for GPUMD to finish..."

# Wait for GPUMD to finish (with timeout)
WAIT_TIMEOUT=300  # 5 minutes
WAITED=0
while kill -0 $GPUMD_PID 2>/dev/null && [ $WAITED -lt $WAIT_TIMEOUT ]; do
    sleep 1
    WAITED=$((WAITED + 1))
done

if kill -0 $GPUMD_PID 2>/dev/null; then
    log_error "GPUMD did not finish within timeout. Killing..."
    kill $GPUMD_PID
    exit 1
fi

log_info ""
log_info "==========================================================="

# Check results
if [ $DRIVER_EXIT -eq 0 ]; then
    log_success "VASP driver completed successfully"
else
    log_error "VASP driver failed with exit code $DRIVER_EXIT"
    log_error "Check driver log: $DRIVER_LOG"
fi

# Save logs before cleanup
log_info "Saving logs to persistent directory..."
save_logs

# Print summary
log_info ""
log_info "Simulation Summary:"
log_info "  GPUMD log:   $LOGDIR/ (saved automatically)"
log_info "  Driver log:  $LOGDIR/ (saved automatically)"
log_info "  Port used:   $PORT"
log_info "  Steps:       $STEPS"
log_info ""
log_info "View logs with:"
log_info "  cat $LOGDIR/gpumd_*.log"
log_info "  cat $LOGDIR/vasp_driver_*.log"

log_success "==========================================================="
log_success "QM/MM simulation completed!"
log_success "==========================================================="

# Exit with driver's exit code
exit $DRIVER_EXIT

