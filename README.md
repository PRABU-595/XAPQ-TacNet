# XAPQ-TacNet: Explainable Adaptive Post-Quantum Cryptography Simulation Framework for Heterogeneous Tactical Networks

**Submission Target:** IEEE WiCOMM-2026 (Track 2: Computing Technologies and Applications)  
**Organization:** Defence Electronics Applications Laboratory (DEAL), DRDO, Dehradun  

---

## 📌 Framework Overview

**XAPQ-TacNet** is a complete, submission-ready Python simulation framework designed to model, analyze, and optimize Post-Quantum Cryptography (PQC) suite selection across heterogeneous military tactical links (VLF, HF, UHF LOS, and Ku-band SATCOM).

The framework implements:
1. **Real-World PQC Benchmarks**: Grounded strictly in NIST FIPS 203/204/205 specifications, `pqm4` ARM Cortex-M4 cycle counts, and Fitzgibbon & Ottaviani (2024) Raspberry Pi 4 TLS handshake data.
2. **Tactical Link Simulator**: Models 4 MIL-STD link classes over 1000 steps, including jamming disruptions and quantum threat level escalations.
3. **Hard Latency Constraint Filter**: Eliminates infeasible (KEM, SIG) pairs prior to decision selection.
4. **LinUCB Contextual Bandit**: Dynamically balances security, latency, and bandwidth using an 8-feature context vector and threat-weighted multi-objective rewards.
5. **Dual-Level XAI Engine**: Derives sub-millisecond operator summary strings and auditor JSON logs directly from LinUCB linear weights without surrogate model overhead.
6. **Baselines & Evaluation**: Static, Random, Rule-Based, and CAAP-Adapted (NSGA-II MOEA via `pymoo`) comparison methods across 30 Monte Carlo random seeds with Mann-Whitney U statistical significance testing.
7. **Commander Trust Study**: Simulates human-in-the-loop decision-making and override quality under opaque vs. explainable AI conditions.

---

## 📁 Repository Structure

```
XAPQ-TacNet/
├── requirements.txt               # Dependencies (numpy, scipy, matplotlib, seaborn, pymoo, pandas)
├── pqc_database.py                # Module 1: NIST FIPS specs & pqm4 benchmarking database
├── tactical_network.py            # Module 2: MIL-STD tactical link dynamic channel simulator
├── feasibility_filter.py          # Module 3: Latency budget hard constraint filter
├── linucb_selector.py             # Module 4: LinUCB contextual bandit algorithm (Li et al., 2010)
├── xai_engine.py                  # Module 5: Sub-millisecond dual-level explanation generator
├── baselines.py                   # Module 6: Static, Random, Rule-Based, & CAAP NSGA-II selectors
├── run_experiment.py              # Module 7: 30-seed simulation driver & CSV exporter
├── analysis_and_plots.py          # Module 8: IEEE conference-quality figures (1-7) & tables (1-3)
├── commander_override_study.py    # Module 9: Commander override and trust simulation
└── README.md                      # Documentation
```

---

## 🚀 Quick Start Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Main Experiment Simulation (30 Seeds x 1000 Steps)
```bash
python run_experiment.py
```
*Outputs `simulation_results.csv` and `audit_log.json`.*

### 3. Generate Publication Figures & Tables
```bash
python analysis_and_plots.py
```
*Generates IEEE vector figures in `figures/` directory and prints Tables 1, 2, and 3 to stdout.*

### 4. Run Commander Trust & Override Study
```bash
python commander_override_study.py
```

---

## 📊 Summary of Main Experimental Results

- **Result 1 (VLF/HF Feasibility Kill-Shot)**: Static policy (`ML-KEM-1024 + ML-DSA-65`) physically fails on VLF/HF links (latency budget exceeded by 270%). XAPQ-TacNet guarantees 100% feasibility.
- **Result 2 (Threat Escalation Adaptation)**: When threat level escalates at step 700, XAPQ-TacNet automatically upgrades high-bandwidth links (UHF/SATCOM) to NIST Level 5 while holding low-bandwidth VLF on Level 1.
- **Result 3 (CAAP Parity + Zero Cost XAI)**: Performs statistically on par with NSGA-II evolutionary optimization ($p > 0.05$ on Mann-Whitney U test) while running in $< 1\text{ ms}$ (vs $> 200\text{ ms}$ for NSGA-II) and providing full explainability.
- **Result 4 (Commander Trust)**: Explainable output reduces unnecessary commander overrides from $30\%$ to $10\%$ while improving override decision quality by $> 85\%$.
