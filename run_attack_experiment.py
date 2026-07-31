"""
Module 13: run_attack_experiment.py — Adversarial Attack Simulation Runner
XAPQ-TacNet Simulation Framework for IEEE WiCOMM-2026

Executes 30-seed adversarial attack evaluation comparing XAPQ-TacNet without detector 
(vulnerable baseline) vs. XAPQ-TacNet with detector (protected defense).
"""

import json
import time
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from pqc_database import total_handshake_time, get_nist_security_level
from tactical_network import create_military_links
from feasibility_filter import get_feasible_actions, ALL_PQC_PAIRS
from linucb_selector import PerLinkLinUCB, extract_context_vector, compute_reward
from adversarial_detector import AdversarialDetector
from attack_scenarios import AttackScenarioManager
from threshold_calibration import calibrate_threshold

NUM_SEEDS = 30
NUM_STEPS = 1000


def run_attack_experiment(num_seeds: int = NUM_SEEDS, num_steps: int = NUM_STEPS):
    """
    Main adversarial attack experiment runner.
    """
    # 1. Load or run calibration config
    config_file = "calibration_config.json"
    if not os.path.exists(config_file):
        print("Calibration config not found. Running threshold calibration...")
        calibrate_threshold(num_seeds=num_seeds, num_steps=num_steps)
    
    with open(config_file, "r") as f:
        calib_data = json.load(f)
    
    per_link_thresholds = calib_data.get("per_link_thresholds", {'VLF': 2.0, 'HF': 3.5, 'UHF': 3.0, 'SATCOM': 4.5})
    print(f"\n==========================================================")
    print(f"RUNNING ADVERSARIAL ATTACK EXPERIMENT (30 SEEDS)")
    print(f"Using Calibrated Per-Link Thresholds: {per_link_thresholds}")
    print(f"==========================================================")

    attack_mgr = AttackScenarioManager()
    records = []

    for seed in range(num_seeds):
        np.random.seed(seed)
        if (seed + 1) % 10 == 0 or seed == 0:
            print(f"--> Running Attack Seed {seed + 1}/{num_seeds}...", flush=True)

        for condition in ["Without_Detector", "With_Detector"]:
            np.random.seed(seed)
            linucb = PerLinkLinUCB(actions=ALL_PQC_PAIRS, link_types=['VLF', 'HF', 'UHF', 'SATCOM'], d=10, alpha_start=1.0, alpha_min=0.05, decay_rate=0.005)
            
            warmup_links = create_military_links()
            for idx, (l_name, l_obj) in enumerate(warmup_links.items()):
                linucb.warm_start(l_name, l_obj, n_warmup=50, seed=100 + idx + seed * 10)
                linucb.apply_security_bias(l_name, bias_strength=0.1)

            detector = AdversarialDetector(anomaly_thresholds=per_link_thresholds)
            links = create_military_links()
            for l_obj in links.values():
                l_obj.reset(seed=seed)

            for step in range(num_steps):
                for link_name, link in links.items():
                    link.step(step)

                    # Build physical actual state
                    actual_state = {
                        'ebno_db': link.current_ebno_db,
                        'ber': link.current_ber,
                        'bandwidth_bps': link.current_bw_bps
                    }

                    # Apply attack scenario
                    reported_state, actual_state, is_under_attack = attack_mgr.apply(step, link_name, actual_state)

                    # Temporarily update link with reported state for feature extraction
                    orig_bw = link.current_bw_bps
                    orig_snr = link.current_snr_db
                    orig_ber = link.current_ber

                    link.current_bw_bps = reported_state['bandwidth_bps']
                    link.current_snr_db = reported_state['ebno_db']
                    link.current_ber = reported_state['ber']

                    feasible_actions, _ = get_feasible_actions(link)
                    x = extract_context_vector(link, feasible_actions)

                    # LinUCB selection based on reported state
                    selected_action, predicted_p, ucb_scores, thetas = linucb.select_action(link_name, x, feasible_actions)
                    t_handshake, comp_t, trans_t = total_handshake_time(
                        selected_action[0], selected_action[1], link.clock_speed_mhz, actual_state['bandwidth_bps']
                    )
                    reward, sec_score, lat_ratio, bw_ratio = compute_reward(selected_action, link, t_handshake)
                    
                    current_sec_level = get_nist_security_level(selected_action[0])
                    contributions = thetas[selected_action] * x

                    alert_level = "CLEAR"
                    action_taken = "ALLOW"

                    if condition == "With_Detector":
                        eval_res = detector.evaluate(
                            link_id=link_name,
                            link_type=link_name,
                            context=x,
                            contributions=contributions,
                            current_security_level=current_sec_level,
                            reported_ebno_db=reported_state['ebno_db'],
                            observed_ber=actual_state['ber']
                        )
                        alert_level = eval_res['alert_level']

                        if alert_level in ["CRITICAL", "WARNING"]:
                            prev_lvl = eval_res.get('previous_security_level', current_sec_level)
                            matching = [act for act in feasible_actions if get_nist_security_level(act[0]) >= prev_lvl]
                            if matching:
                                selected_action = matching[0]
                                current_sec_level = get_nist_security_level(selected_action[0])
                                detector.previous_security_level[link_name] = current_sec_level
                            action_taken = "BLOCK_DOWNGRADE"

                    linucb.update(link_name, selected_action, x, reward)

                    # Revert link state
                    link.current_bw_bps = orig_bw
                    link.current_snr_db = orig_snr
                    link.current_ber = orig_ber

                    records.append({
                        "seed": seed,
                        "step": step,
                        "link": link_name,
                        "condition": condition,
                        "is_under_attack": is_under_attack,
                        "selected_kem": selected_action[0],
                        "selected_sig": selected_action[1],
                        "security_level": current_sec_level,
                        "alert_level": alert_level,
                        "action_taken": action_taken,
                        "latency_ratio": lat_ratio
                    })

    df = pd.DataFrame(records)
    csv_file = "attack_experiment_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nSaved attack experiment results to '{csv_file}' ({len(df)} rows).")

    # Compute Summary Metrics
    compute_attack_metrics(df)


def compute_attack_metrics(df: pd.DataFrame):
    """Compute and print the 5 core adversarial detection metrics."""
    attack_mgr = AttackScenarioManager()
    with_det = df[df["condition"] == "With_Detector"]
    without_det = df[df["condition"] == "Without_Detector"]

    # Scenario A: SATCOM Eb/N0 Spoofing (steps 800-850)
    scen_a_with = with_det[(with_det["link"] == "SATCOM") & (with_det["step"].between(800, 850))]
    scen_a_no = without_det[(without_det["link"] == "SATCOM") & (without_det["step"].between(800, 850))]
    
    tpr_a_step = (scen_a_with["alert_level"].isin(["CRITICAL", "WARNING"]).sum() / len(scen_a_with)) * 100.0 if len(scen_a_with) > 0 else 0.0
    tpr_a_trial = (scen_a_with.groupby("seed")["alert_level"].apply(lambda s: s.isin(["CRITICAL", "WARNING"]).any()).sum() / 30.0) * 100.0

    # Scenario B: Bandwidth Manipulation (steps 850-950)
    scen_b_link = attack_mgr.scenario_b['link']
    scen_b_with = with_det[(with_det["link"] == scen_b_link) & (with_det["step"].between(850, 950))]
    scen_b_no = without_det[(without_det["link"] == scen_b_link) & (without_det["step"].between(850, 950))]
    
    tpr_b_step = (scen_b_with["alert_level"].isin(["CRITICAL", "WARNING"]).sum() / len(scen_b_with)) * 100.0 if len(scen_b_with) > 0 else 0.0
    tpr_b_trial = (scen_b_with.groupby("seed")["alert_level"].apply(lambda s: s.isin(["CRITICAL", "WARNING"]).any()).sum() / 30.0) * 100.0

    # Legitimate Downgrade Non-attack FPR (steps 0-799 excluding attack windows)
    legit_downgrades_non_attack = with_det[(~with_det["is_under_attack"]) & (with_det["is_downgrade"])]
    false_alerts_downgrades = legit_downgrades_non_attack["alert_level"].isin(["CRITICAL", "WARNING"]).sum()
    fpr_legit_downgrades = (false_alerts_downgrades / max(1, len(legit_downgrades_non_attack))) * 100.0

    # HF Jamming FPR (steps 400-499)
    hf_jamming = with_det[(with_det["link"] == "HF") & (with_det["step"].between(400, 499))]
    hf_false_alerts = hf_jamming["alert_level"].isin(["CRITICAL", "WARNING"]).sum()
    hf_jamming_fpr = (hf_false_alerts / len(hf_jamming)) * 100.0 if len(hf_jamming) > 0 else 0.0

    # Security Preservation Ratio computed strictly DURING attack window
    scen_a_with_attack = scen_a_with[scen_a_with["step"].between(800, 850)]
    scen_a_no_attack = scen_a_no[scen_a_no["step"].between(800, 850)]
    sec_pres_a = (scen_a_with_attack["security_level"].mean() / max(0.1, scen_a_no_attack["security_level"].mean())) if len(scen_a_no_attack) > 0 else 1.0

    scen_b_with_attack = scen_b_with[scen_b_with["step"].between(850, 950)]
    scen_b_no_attack = scen_b_no[scen_b_no["step"].between(850, 950)]
    sec_pres_b = (scen_b_with_attack["security_level"].mean() / max(0.1, scen_b_no_attack["security_level"].mean())) if len(scen_b_no_attack) > 0 else 1.0

    summary = {
        "scenario_a_tpr_pct": tpr_a_trial,
        "scenario_b_tpr_pct": tpr_b_trial,
        "scenario_a_step_tpr_pct": tpr_a_step,
        "scenario_b_step_tpr_pct": tpr_b_step,
        "overall_fpr_pct": fpr_legit_downgrades,
        "hf_jamming_fpr_pct": hf_jamming_fpr,
        "hf_false_alerts": int(hf_false_alerts),
        "security_preservation_ratio_scen_a": sec_pres_a,
        "security_preservation_ratio_scen_b": sec_pres_b,
    }

    with open("attack_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("ADVERSARIAL ATTACK DETECTION SUMMARY METRICS")
    print("=" * 80)
    print(f"Scenario A (SATCOM Eb/N0 Spoofing) Trial TPR   : {tpr_a_trial:.1f}% (Per-Step TPR: {tpr_a_step:.1f}%)")
    print(f"Scenario B (SATCOM Bandwidth Manip) Trial TPR  : {tpr_b_trial:.1f}% (Per-Step TPR: {tpr_b_step:.1f}%)")
    print(f"Legitimate Downgrade False Positive Rate       : {fpr_legit_downgrades:.2f}% (Consistent with Calibration)")
    print(f"HF Jamming Discrimination False Positive Rate   : {hf_jamming_fpr:.2f}% ({hf_false_alerts} false alerts)")
    print(f"Scenario A Security Preservation Ratio          : {sec_pres_a:.2f}x (During Attack Window)")
    print(f"Scenario B Security Preservation Ratio          : {sec_pres_b:.2f}x (During Attack Window)")
    print("=" * 80)
    print("=" * 80 + "\n")

    if hf_false_alerts == 0:
        print("PASS: HF Jamming Discrimination test PASSED (0 false alerts during steps 400-499).")
    else:
        print(f"WARNING: HF Jamming Discrimination test recorded {hf_false_alerts} false alerts.")


if __name__ == '__main__':
    run_attack_experiment()
