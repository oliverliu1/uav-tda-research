#!/bin/bash

echo "Starting Script 3 at $(date)..." | tee -a pipeline_run.log
python -u scripts/03_tda_c2_manifold.py 2>&1 | tee -a pipeline_run.log
echo "Script 3 complete: $(date)" | tee -a pipeline_run.log

echo "Starting Script 4 at $(date)..." | tee -a pipeline_run.log
python -u scripts/04_tda_network_manifold.py 2>&1 | tee -a pipeline_run.log
echo "Script 4 complete: $(date)" | tee -a pipeline_run.log

echo "Starting Script 5 at $(date)..." | tee -a pipeline_run.log
python -u scripts/05_tda_physical_manifold.py 2>&1 | tee -a pipeline_run.log
echo "Script 5 complete: $(date)" | tee -a pipeline_run.log

echo "Starting Script 6 at $(date)..." | tee -a pipeline_run.log
python -u scripts/06_tda_features_extraction.py 2>&1 | tee -a pipeline_run.log
echo "Script 6 complete: $(date)" | tee -a pipeline_run.log

echo "Starting Script 7 at $(date)..." | tee -a pipeline_run.log
python -u scripts/07_tda_enhanced_models.py 2>&1 | tee -a pipeline_run.log
echo "Script 7 complete: $(date)" | tee -a pipeline_run.log

echo "Starting Script 8 at $(date)..." | tee -a pipeline_run.log
python -u scripts/08_comparative_analysis.py 2>&1 | tee -a pipeline_run.log
echo "Pipeline complete: $(date)" | tee -a pipeline_run.log

echo "PIPELINE COMPLETE!"
