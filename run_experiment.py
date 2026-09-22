"""
Module 7: run_experiment.py — Main Experiment Runner
XAPQ-TacNet Simulation Framework for IEEE WiCOMM-2026

Executes 30 independent Monte Carlo seeds over 1000 simulation steps across 4 tactical 
links and 5 selection strategies. Exports full raw metric logs to CSV.
"""

import time
import os
import csv
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from pqc_database import total_handshake_time, get_nist_security_level
from tactical_network import create_military_links
from feasibility_filter import get_feasible_actions, ALL_PQC_PAIRS
from linucb_selector import PerLinkLinUCB, extract_context_vector, compute_reward
from xai_engine import XAIEngine
from baselines import StaticSelector, RandomSelector, RuleBasedSelector, ExhaustiveOracleSelector

METHODS = ["XAPQ_TacNet", "Static", "Random", "RuleBased", "Exhaustive_Oracle"]
NUM_SEEDS = 30
NUM_STEPS = 1000


def run_full_experiment(num_seeds: int = NUM_SEEDS, num_steps: int = NUM_STEPS, csv_filename: str = "simulation_results.csv"):
    """
    Main simulation execution loop.
    Runs Monte Carlo simulation over specified random seeds and steps.
    """
    print(f"Starting XAPQ-TacNet Experiment Simulation:", flush=True)
    print(f"  Seeds: {num_seeds}", flush=True)
    print(f"  Steps per Seed: {num_steps}", flush=True)
    print(f"  Methods: {METHODS}", flush=True)
    print(f"  Links: VLF, HF, UHF, SATCOM", flush=True)
    print("=" * 60, flush=True)

    t_experiment_start = time.time()
    records = []

    xai_engine = XAIEngine()

    for seed in range(num_seeds):
        np.random.seed(seed)
        print(f"--> Running Seed {seed + 1}/{num_seeds}...", flush=True)

        # Instantiate separate selectors per seed
        linucb = PerLinkLinUCB(actions=ALL_PQC_PAIRS, link_types=['VLF', 'HF', 'UHF', 'SATCOM'], d=10, alpha_start=1.0, alpha_min=0.05, decay_rate=0.005)
        
        # Warm-start each link type and apply optimistic security bias
        warmup_links = create_military_links()
        for idx, (l_name, l_obj) in enumerate(warmup_links.items()):
            linucb.warm_start(l_name, l_obj, n_warmup=50, seed=100 + idx + seed * 10)
            linucb.apply_security_bias(l_name, bias_strength=0.1)

        static_sel = StaticSelector()
        random_sel = RandomSelector()
        rule_sel = RuleBasedSelector()
        oracle_sel = ExhaustiveOracleSelector()

        links = create_military_links()
        for l_obj in links.values():
            l_obj.reset(seed=seed)

        for step in range(num_steps):
            for link_name, link in links.items():
                # 1. Advance physical channel dynamics
                link.step(step)

                # 2. Extract context vector & obtain feasible actions
                feasible_actions, _ = get_feasible_actions(link)
                x = extract_context_vector(link, feasible_actions)

                # Find ground-truth optimal action among feasible set for adaptation_correct evaluation
                best_feasible_reward = -float("inf")
                optimal_action = feasible_actions[0]
                for act in feasible_actions:
                    t_time, _, _ = total_handshake_time(act[0], act[1], link.clock_speed_mhz, link.current_bw_bps)
                    r_val, _, _, _ = compute_reward(act, link, t_time)
                    if r_val > best_feasible_reward:
                        best_feasible_reward = r_val
                        optimal_action = act

                # 3. Evaluate each method
                for method in METHODS:
                    explanation_time_us = 0.0
                    
                    if method == "XAPQ_TacNet":
                        selected_action, predicted_p, ucb_scores, thetas = linucb.select_action(link_name, x, feasible_actions)
                        t_handshake, comp_t, trans_t = total_handshake_time(
                            selected_action[0], selected_action[1], link.clock_speed_mhz, link.current_bw_bps
                        )
                        reward, sec_score, lat_ratio, bw_ratio = compute_reward(selected_action, link, t_handshake)
                        
                        # Update LinUCB online weights
                        linucb.update(link_name, selected_action, x, reward)

                        # Generate XAI explanations for sample steps
                        op_summary, aud_log = xai_engine.generate_explanation(
                            step_number=step,
                            link_name=link_name,
                            x=x,
                            feasible_actions=feasible_actions,
                            selected_action=selected_action,
                            theta_selected=thetas[selected_action],
                            all_thetas=thetas,
                            predicted_reward=predicted_p,
                            actual_reward=reward,
                            threat_level=link.threat_level
                        )
                        explanation_time_us = aud_log["explanation_generation_time_us"]

                    elif method == "Static":
                        selected_action = static_sel.select_action(feasible_actions)
                        t_handshake, comp_t, trans_t = total_handshake_time(
                            selected_action[0], selected_action[1], link.clock_speed_mhz, link.current_bw_bps
                        )
                        reward, sec_score, lat_ratio, bw_ratio = compute_reward(selected_action, link, t_handshake)

                    elif method == "Random":
                        selected_action = random_sel.select_action(feasible_actions)
                        t_handshake, comp_t, trans_t = total_handshake_time(
                            selected_action[0], selected_action[1], link.clock_speed_mhz, link.current_bw_bps
                        )
                        reward, sec_score, lat_ratio, bw_ratio = compute_reward(selected_action, link, t_handshake)

                    elif method == "RuleBased":
                        selected_action = rule_sel.select_action(link, feasible_actions)
                        t_handshake, comp_t, trans_t = total_handshake_time(
                            selected_action[0], selected_action[1], link.clock_speed_mhz, link.current_bw_bps
                        )
                        reward, sec_score, lat_ratio, bw_ratio = compute_reward(selected_action, link, t_handshake)

                    elif method == "Exhaustive_Oracle":
                        selected_action = oracle_sel.select_action(link, feasible_actions)
                        t_handshake, comp_t, trans_t = total_handshake_time(
                            selected_action[0], selected_action[1], link.clock_speed_mhz, link.current_bw_bps
                        )
                        reward, sec_score, lat_ratio, bw_ratio = compute_reward(selected_action, link, t_handshake)

                    # Compute adaptation accuracy
                    adaptation_correct = 1.0 if (selected_action == optimal_action) else 0.0

                    # Check budget compliance
                    is_infeasible = 1.0 if (t_handshake > link.latency_budget_sec) else 0.0

                    records.append({
                        "seed": seed,
                        "step": step,
                        "link": link_name,
                        "method": method,
                        "kem": selected_action[0],
                        "sig": selected_action[1],
                        "security_score": sec_score,
                        "handshake_time_sec": t_handshake,
                        "latency_budget_sec": link.latency_budget_sec,
                        "latency_ratio": lat_ratio,
                        "bandwidth_ratio": bw_ratio,
                        "reward": reward,
                        "adaptation_correct": adaptation_correct,
                        "is_infeasible": is_infeasible,
                        "explanation_time_us": explanation_time_us,
                        "snr_db": link.current_snr_db,
                        "bandwidth_bps": link.current_bw_bps,
                        "threat_level": link.threat_level
                    })

    t_experiment_end = time.time()
    elapsed_min = (t_experiment_end - t_experiment_start) / 60.0
    print(f"\nExperiment execution complete in {elapsed_min:.2f} minutes.", flush=True)

    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(csv_filename, index=False)
    print(f"Saved full simulation results to '{csv_filename}' ({len(df)} rows).", flush=True)

    # Save audit log
    xai_engine.save_audit_log("audit_log.json")
    print("Saved audit log to 'audit_log.json'.", flush=True)

    return df


if __name__ == "__main__":
    import sys
    if "--attack" in sys.argv:
        from run_attack_experiment import run_attack_experiment
        run_attack_experiment(num_seeds=30, num_steps=1000)
    else:
        run_full_experiment(num_seeds=30, num_steps=1000)
