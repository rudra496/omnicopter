# Distilling Energy-Aware Null-Space Control for an Omnidirectional UAV into a Real-Time Supervised Oracle

**Author:** Rudra Sarker  
**Email:** rudrasarker130@gmail.com  
**Repository:** https://github.com/rudra496/omnicopter

This repository contains the official implementation, dataset, and supplementary materials for the paper:

> **"Distilling Energy-Aware Null-Space Control for an Omnidirectional UAV into a Real-Time Supervised Oracle"**

## 📋 Overview

We propose a two-stage learning framework for energy-aware null-space control of omnidirectional multirotor aerial vehicles (OMAVs):

1. **Stage 1:** Train an energy-aware RL expert using Soft Actor-Critic (SAC) under stochastic wind and domain randomization
2. **Stage 2:** Distill the expert into a lightweight XGBoost oracle for real-time deployment

Key results:
- Distilled oracle achieves \(R^2 = 0.9918\) and \(0.9947\) fidelity
- Inference latency: 0.0569 ms/sample (suitable for embedded flight computers)
- Energy savings up to 32% compared to pseudo-inverse baseline

## 🗂️ Repository Structure

```
omnicopter/                           # Root folder of your repository
│
├── 📜 README.md                      # Main project documentation
├── 📜 CITATION.cff                   # Citation metadata file
├── 📜 requirements.txt               # Python dependencies
├── 📜 .gitignore                     # Files to ignore in Git
│
├── 📂 paper/                         # All paper-related materials
│   ├── 📜 main.tex                   # LaTeX source (provided by you)
│   ├── 📜 references.bib             # Bibliography (provided by you)
│   ├── 📜 omnicopter.pdf             # Compiled PDF (provided by you)
│   └── 📂 figures/                   # Your 8 PNG figures go here
│       ├── 📜 fig1_wind_distribution.png
│       ├── 📜 fig2_power_vs_wind.png
│       ├── 📜 fig3_savings_violin.png
│       ├── 📜 fig4_nullspace_map.png
│       ├── 📜 fig5_corr_heatmap.png
│       ├── 📜 fig6_oracle_scatter.png
│       ├── 📜 fig7_robustness_curve.png
│       └── 📜 fig8_pca_savings.png
│
├── 📂 src/                           # Source code
│   ├── 📂 rl/                        # Reinforcement learning code
│   │   ├── 📜 __init__.py
│   │   ├── 📜 train_sac.py           # SAC training script
│   │   └── 📜 expert_policy.py       # Frozen expert policy
│   │
│   ├── 📂 xgboost/                   # XGBoost distillation code
│   │   ├── 📜 __init__.py
│   │   ├── 📜 distill.py             # Distillation script
│   │   └── 📜 oracle_model.py        # XGBoost oracle
│   │
│   └── 📂 utils/                     # Utility functions
│       ├── 📜 __init__.py
│       ├── 📜 allocation.py          # Null-space allocation utilities
│       ├── 📜 energy_proxy.py        # Power proxy calculation
│       └── 📜 data_loader.py         # Dataset loading utilities
│
├── 📂 notebooks/                     # Jupyter notebooks for analysis
│   ├── 📜 01_dataset_exploration.ipynb
│   ├── 📜 02_energy_analysis.ipynb
│   ├── 📜 03_oracle_evaluation.ipynb
│   └── 📜 04_visualization.ipynb
│
├── 📂 configs/                       # Configuration files
│   ├── 📜 sac_params.yaml           # SAC hyperparameters
│   ├── 📜 xgboost_params.yaml       # XGBoost hyperparameters
│   └── 📜 env_config.json           # Environment parameters
│
├── 📂 data/                          # Data directory
│   ├── 📂 raw/                       # Raw data (not in Git)
│   │   ├── 📜 omav_sac_expert.zip   # Your ZIP file goes here
│   │   └── 📜 .gitkeep              # Keep folder in Git
│   │
│   ├── 📂 processed/                 # Processed data (in Git)
│   │   ├── 📜 table1_statistics.csv # Your 4 CSV tables go here
│   │   ├── 📜 table2_energy_modes.csv
│   │   ├── 📜 table3_oracle_perf.csv
│   │   ├── 📜 table4_robustness.csv
│   │   └── 📜 .gitkeep
│   │
│   └── 📜 README_DATA.md             # Data documentation
│
├── 📂 results/                       # Experiment results
│   ├── 📂 logs/                      # Training logs
│   ├── 📂 models/                    # Saved models
│   └── 📜 .gitkeep
│
└── 📂 tests/                         # Unit tests
    ├── 📜 test_allocation.py
    ├── 📜 test_energy_proxy.py
    └── 📜 test_data_loader.py
```

## 📊 Data Availability

### Large Dataset (300+ MB)
The primary frozen-expert dataset (`dataset_rl_distill.csv`) is **hosted separately** due to GitHub file size limitations.

**Download instructions:**
1. Go to the [Releases](https://github.com/rudra496/omnicopter/releases) section of this repository
2. Download the latest dataset file from the release assets
3. Place it in your local `data/raw/` directory for use with the code

### Files Included in Repository
- **8 PNG figures** in `paper/figures/` for all paper visualizations
- **4 CSV tables** in `data/processed/` with experimental results
- **Expert data ZIP** (`omav_sac_expert.zip`) in `data/raw/`

## 🚀 Quick Start

### Prerequisites
```bash
# Clone repository
git clone https://github.com/rudra496/omnicopter.git
cd omnicopter

# Install dependencies
pip install -r requirements.txt
```

### Running the Code

**Train SAC Expert:**
```bash
python src/rl/train_sac.py --config configs/sac_params.yaml
```

**Distill to XGBoost Oracle:**
```bash
python src/xgboost/distill.py \
  --data data/processed/ \
  --config configs/xgboost_params.yaml
```

### Analysis Notebooks

Open Jupyter notebooks in the `notebooks/` directory for:
- Dataset exploration and statistics
- Energy savings analysis
- Oracle performance evaluation
- Visualization generation

## 📈 Results Summary

| Metric | Value |
|--------|-------|
| Oracle fidelity (R² for z₁) | 0.9918 |
| Oracle fidelity (R² for z₂) | 0.9947 |
| Inference latency | 0.0569 ms/sample |
| Maximum energy savings | 32% at 15 m/s wind |
| Dataset samples | 200,000 |

## 🔧 Configuration

Key configuration files:
- `configs/sac_params.yaml`: SAC training hyperparameters
- `configs/xgboost_params.yaml`: XGBoost distillation settings
- `configs/env_config.json`: Simulation environment parameters

## 📄 Paper

The complete paper is available in multiple formats:
- `paper/main.tex`: LaTeX source
- `paper/omnicopter.pdf`: Compiled PDF
- All figures in `paper/figures/`

## 🤝 Contributing

This is a research repository. For questions or issues:
1. Check existing issues in the GitHub repository
2. Email the author at rudrasarker130@gmail.com

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.