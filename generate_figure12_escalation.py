"""
generate_figure12_escalation.py — Figure 12: Threat Escalation Response
XAPQ-TacNet Simulation Framework for IEEE WiCOMM-2026

Generates Figure 12 showing focused threat escalation adaptation on SATCOM link from 
step 600 to step 1000, with clean non-overlapping annotations and physical scoping text.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure IEEE Matplotlib Style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8.5

def generate_figure12(csv_filepath: str = "simulation_results.csv"):
    if not os.path.exists(csv_filepath):
        print(f"Error: '{csv_filepath}' not found.")
        return

    df = pd.read_csv(csv_filepath)
    sat_df = df[(df["link"] == "SATCOM") & (df["method"] == "XAPQ_TacNet") & (df["step"].between(600, 1000))]
    step_means = sat_df.groupby("step")["security_score"].mean().reset_index()
    step_means["security_level"] = step_means["security_score"] * 5.0

    pre_means = step_means[step_means["step"].between(600, 699)]["security_level"].mean()
    post_means = step_means[step_means["step"].between(705, 1000)]["security_level"].mean()

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)

    # Plot security level line
    ax.plot(step_means["step"], step_means["security_level"], color='#d62728', lw=2.0, label='XAPQ-TacNet Security Level (SATCOM)')
    
    # Threat escalation line
    ax.axvline(x=700, color='purple', linestyle='--', lw=1.8, label='Threat Escalation Event (Step 700)')

    # Mean lines
    ax.axhline(y=pre_means, xmin=0.0, xmax=0.25, color='gray', linestyle=':', lw=1.2)
    ax.axhline(y=post_means, xmin=0.27, xmax=1.0, color='gray', linestyle=':', lw=1.2)

    # Pre-escalation callout box (placed safely below line)
    ax.annotate(
        f'Pre-Escalation Mean: {pre_means:.2f}\n(NIST Security Level 5)',
        xy=(650, pre_means), xytext=(610, 3.5),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
        fontsize=8, fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9)
    )

    # Post-escalation callout box (placed safely below line to right)
    ax.annotate(
        f'Post-Escalation Mean: {post_means:.2f}\n(Maintains Level 5 Protection)',
        xy=(850, post_means), xytext=(770, 3.5),
        arrowprops=dict(facecolor='purple', shrink=0.05, width=1, headwidth=5),
        fontsize=8, fontweight='bold', color='purple',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='purple', alpha=0.9)
    )

    # Physical scoping note text box (placed cleanly at bottom)
    ax.text(
        0.5, 0.12,
        "Physical Scoping Note: On bandwidth-constrained links (VLF, HF), security level\n"
        "remains locked at maximum physically feasible level (Level 1) — higher algorithms\n"
        "are physically infeasible due to extreme bandwidth limits regardless of threat level.",
        transform=ax.transAxes, ha='center', fontsize=7.5, fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.95)
    )

    ax.set_title("Figure 12: Focused Threat Escalation Response on SATCOM Link")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("NIST Security Level")
    ax.set_ylim(0.5, 6.2)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.95)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/figure12_threat_escalation.pdf")
    plt.savefig("figures/figure12_threat_escalation.png")
    plt.close()
    print("Regenerated Figure 12: Threat Escalation Response in 'figures/figure12_threat_escalation.pdf/png'.")

if __name__ == '__main__':
    generate_figure12()
