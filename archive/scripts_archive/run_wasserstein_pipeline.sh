#!/bin/bash
################################################################################
# run_wasserstein_pipeline.sh
#
# Runs the complete Wasserstein approach pipeline sequentially.
# Estimated total time: 9-13 hours (mostly Script 3W)
#
# Usage:
#   chmod +x scripts/run_wasserstein_pipeline.sh
#   nohup scripts/run_wasserstein_pipeline.sh > logs/wasserstein_pipeline.log 2>&1 &
#
# Author: Oliver Liu
# Date: April 2026
################################################################################

# Exit on error
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Log file
MASTER_LOG="logs/wasserstein_pipeline_$(date +%Y%m%d_%H%M%S).log"

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG"
}

# Function to print colored messages
print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    log "$1"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    log "✓ $1"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
    log "✗ ERROR: $1"
}

print_warning() {
    echo -e "${YELLOW}⚠  $1${NC}"
    log "⚠  $1"
}

################################################################################
# START PIPELINE
################################################################################

print_header "WASSERSTEIN APPROACH PIPELINE - STARTING"
log "Project directory: $PROJECT_DIR"
log "Master log file: $MASTER_LOG"
echo ""

# Record start time
PIPELINE_START=$(date +%s)

################################################################################
# SCRIPT 2W: Baseline Barcodes (5-10 minutes)
################################################################################

print_header "STEP 1/5: Computing Baseline Barcodes (Script 2W)"
log "Estimated time: 5-10 minutes"
echo ""

STEP_START=$(date +%s)

if python scripts/2W_wasserstein_baseline.py 2>&1 | tee -a "$MASTER_LOG"; then
    STEP_END=$(date +%s)
    STEP_DURATION=$((STEP_END - STEP_START))
    print_success "Script 2W complete in $((STEP_DURATION / 60)) minutes $((STEP_DURATION % 60)) seconds"
    echo ""
else
    print_error "Script 2W failed"
    exit 1
fi

################################################################################
# SCRIPT 3W: Per-Flow Persistence (8-12 hours)
################################################################################

print_header "STEP 2/5: Computing Per-Flow Persistence (Script 3W)"
log "Estimated time: 8-12 hours"
print_warning "This is the longest step - great time for a break!"
echo ""

STEP_START=$(date +%s)

if python scripts/3W_wasserstein_per_flow.py 2>&1 | tee -a "$MASTER_LOG"; then
    STEP_END=$(date +%s)
    STEP_DURATION=$((STEP_END - STEP_START))
    print_success "Script 3W complete in $((STEP_DURATION / 3600)) hours $((STEP_DURATION % 3600 / 60)) minutes"
    echo ""
else
    print_error "Script 3W failed"
    exit 1
fi

################################################################################
# SCRIPT 4W: Wasserstein Distances (30-60 minutes)
################################################################################

print_header "STEP 3/5: Computing Wasserstein Distances (Script 4W)"
log "Estimated time: 30-60 minutes"
echo ""

STEP_START=$(date +%s)

if python scripts/4W_wasserstein_distances.py 2>&1 | tee -a "$MASTER_LOG"; then
    STEP_END=$(date +%s)
    STEP_DURATION=$((STEP_END - STEP_START))
    print_success "Script 4W complete in $((STEP_DURATION / 60)) minutes $((STEP_DURATION % 60)) seconds"
    echo ""
else
    print_error "Script 4W failed"
    exit 1
fi

################################################################################
# SCRIPT 5W: Z-Score & Detection (10 minutes)
################################################################################

print_header "STEP 4/5: Z-Score Normalization & Detection (Script 5W)"
log "Estimated time: 10 minutes"
echo ""

STEP_START=$(date +%s)

if python scripts/5W_wasserstein_detection.py 2>&1 | tee -a "$MASTER_LOG"; then
    STEP_END=$(date +%s)
    STEP_DURATION=$((STEP_END - STEP_START))
    print_success "Script 5W complete in $((STEP_DURATION / 60)) minutes $((STEP_DURATION % 60)) seconds"
    echo ""
else
    print_error "Script 5W failed"
    exit 1
fi

################################################################################
# SCRIPT 6W: Evaluation & Visualization (15 minutes)
################################################################################

print_header "STEP 5/5: Evaluation & Visualization (Script 6W)"
log "Estimated time: 15 minutes"
echo ""

STEP_START=$(date +%s)

if python scripts/6W_wasserstein_evaluation.py 2>&1 | tee -a "$MASTER_LOG"; then
    STEP_END=$(date +%s)
    STEP_DURATION=$((STEP_END - STEP_START))
    print_success "Script 6W complete in $((STEP_DURATION / 60)) minutes $((STEP_DURATION % 60)) seconds"
    echo ""
else
    print_error "Script 6W failed"
    exit 1
fi

################################################################################
# PIPELINE COMPLETE
################################################################################

PIPELINE_END=$(date +%s)
TOTAL_DURATION=$((PIPELINE_END - PIPELINE_START))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$((TOTAL_DURATION % 3600 / 60))

print_header "WASSERSTEIN PIPELINE COMPLETE!"
echo ""
print_success "Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
print_success "All scripts executed successfully"
echo ""

log "Pipeline completed successfully"
log "Total duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
log "End time: $(date +'%Y-%m-%d %H:%M:%S')"

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}🎉 SUCCESS! Both approaches complete:${NC}"
echo -e "${GREEN}  ✓ Supervised (Scripts 1-8)${NC}"
echo -e "${GREEN}  ✓ Unsupervised (Scripts 2W-6W)${NC}"
echo -e "${GREEN}${NC}"
echo -e "${GREEN}Results ready for poster presentation!${NC}"
echo -e "${GREEN}================================================================================${NC}"

exit 0
