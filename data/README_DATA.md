# Data Directory

## Overview
This directory contains all data files for the OmniCopter project.

## Structure

### `raw/` - Raw Data Files
- **Large files stored outside Git** (use GitHub Releases)
- `dataset_rl_distill.csv`: Primary 300+ MB dataset (download from Releases)
- `omav_sac_expert.zip`: Compressed expert data

### `processed/` - Processed Data Files
- **Small files stored in Git**
- `table1_statistics.csv`: Dataset statistics table
- `table2_energy_modes.csv`: Energy statistics by flight mode
- `table3_oracle_perf.csv`: Oracle performance metrics
- `table4_robustness.csv`: Robustness analysis results

## Dataset Details

### Primary Dataset (`dataset_rl_distill.csv`)
- **Size:** 300+ MB
- **Samples:** 200,000
- **Columns:**
  - 21 observation dimensions
  - 2 null-space coefficients (z1, z2)
  - Environment parameters (wind, mass, inertia, thrust coefficient)
  - Power metrics for baseline and expert

### Expert Data (`omav_sac_expert.zip`)
- Contains frozen SAC expert policy
- Training logs and checkpoints
- Environment configurations

## Usage
See the main README for download instructions for large files.