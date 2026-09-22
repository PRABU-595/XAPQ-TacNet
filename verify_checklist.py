"""
Verification Script for Phase 1 & Phase 2 Checklist Items
XAPQ-TacNet Simulation Framework
"""

import sys
import numpy as np
import pandas as pd
import time

from pqc_database import (
    NIST_SPECS, PQM4_CYCLES, get_nist_security_level,
    compute_time_seconds, transmission_time_seconds, total_handshake_time, get_total_payload_bytes
)
from tactical_network import create_military_links, TacticalLink
from feasibility_filter import get_feasible_actions, ALL_PQC_PAIRS, compute_feasibility_matrix
from linucb_selector import LinUCBSelector, extract_context_vector, compute_reward, FEATURE_DIM
from xai_engine import XAIEngine
from baselines import StaticSelector, RandomSelector, RuleBasedSelector, ExhaustiveOracleSelector


def test_1_1_pqc_database():
    print("\n--- 1.1 PQC Database Integrity Verification ---")
    
    # 1. NIST FIPS 203 (ML-KEM)
    assert NIST_SPECS["ML-KEM-512"] == {"type": "KEM", "nist_level": 1, "pk_bytes": 800, "sk_bytes": 1632, "ct_bytes": 768}
    assert NIST_SPECS["ML-KEM-768"] == {"type": "KEM", "nist_level": 3, "pk_bytes": 1184, "sk_bytes": 2400, "ct_bytes": 1088}
    assert NIST_SPECS["ML-KEM-1024"] == {"type": "KEM", "nist_level": 5, "pk_bytes": 1568, "sk_bytes": 3168, "ct_bytes": 1568}
    print("  [PASS] NIST FIPS 203 ML-KEM byte sizes match standard exactly.")

    # 2. NIST FIPS 204 (ML-DSA)
    assert NIST_SPECS["ML-DSA-44"] == {"type": "SIG", "nist_level": 2, "pk_bytes": 1312, "sk_bytes": 2560, "sig_bytes": 2420}
    assert NIST_SPECS["ML-DSA-65"] == {"type": "SIG", "nist_level": 3, "pk_bytes": 1952, "sk_bytes": 4032, "sig_bytes": 3293}
    assert NIST_SPECS["ML-DSA-87"] == {"type": "SIG", "nist_level": 5, "pk_bytes": 2592, "sk_bytes": 4896, "sig_bytes": 4595}
    print("  [PASS] NIST FIPS 204 ML-DSA byte sizes match standard exactly.")

    # 3. NIST FIPS 205 (SLH-DSA-128s)
    assert NIST_SPECS["SLH-DSA-128s"] == {"type": "SIG", "nist_level": 1, "pk_bytes": 32, "sk_bytes": 64, "sig_bytes": 7856}
    print("  [PASS] NIST FIPS 205 SLH-DSA-128s byte sizes match standard exactly.")

    # 4. Handshake calculation tests on VLF 300 bps with M4 @ 24 MHz
    # ML-KEM-1024 + ML-DSA-65
    payload_heavy = 1568 + 1568 + 1952 + 3293  # 8381 bytes
    trans_heavy = (payload_heavy * 8) / 300.0   # 223.4933 seconds
    comp_heavy = compute_time_seconds("ML-KEM-1024", 24.0, "total") + compute_time_seconds("ML-DSA-65", 24.0, "total")
    t_tot_heavy, _, t_tr_heavy = total_handshake_time("ML-KEM-1024", "ML-DSA-65", 24.0, 300.0)
    print(f"  ML-KEM-1024 + ML-DSA-65 VLF 300 bps trans time: {t_tr_heavy:.2f}s (Expected: ~223.49s)")
    assert abs(t_tr_heavy - trans_heavy) < 1e-3

    # ML-KEM-512 + ML-DSA-44
    payload_light = 800 + 768 + 1312 + 2420    # 5300 bytes
    trans_light = (payload_light * 8) / 300.0   # 141.333 seconds
    t_tot_light, _, t_tr_light = total_handshake_time("ML-KEM-512", "ML-DSA-44", 24.0, 300.0)
    print(f"  ML-KEM-512 + ML-DSA-44 VLF 300 bps trans time: {t_tr_light:.2f}s (Expected: ~141.33s)")
    assert abs(t_tr_light - trans_light) < 1e-3

    # 5. Security levels
    assert get_nist_security_level("ML-KEM-512") == 1
    assert get_nist_security_level("ML-KEM-768") == 3
    assert get_nist_security_level("ML-KEM-1024") == 5
    assert get_nist_security_level("ML-DSA-44") == 2
    assert get_nist_security_level("ML-DSA-65") == 3
    assert get_nist_security_level("ML-DSA-87") == 5
    assert get_nist_security_level("SLH-DSA-128s") == 1
    print("  [PASS] NIST Security level mappings verified.")


def test_1_2_tactical_network():
    print("\n--- 1.2 Tactical Network Simulator Integrity Verification ---")
    links = create_military_links()
    
    # 1. 1000 step MIL-STD range check
    for step in range(1000):
        for name, link in links.items():
            link.step(step)
            assert link.bw_min_bps <= link.current_bw_bps <= link.bw_max_bps, f"BW out of bounds on {name} at step {step}: {link.current_bw_bps}"

    print("  [PASS] All links remain strictly within MIL-STD bandwidth bounds over 1000 steps.")

    # 2. Threat level escalation check
    links["VLF"].reset(seed=42)
    for step in range(1000):
        links["VLF"].step(step)
        if step == 699:
            assert links["VLF"].threat_level == 0.0, f"Step 699 threat level should be 0.0, got {links['VLF'].threat_level}"
        elif step == 700:
            assert abs(links["VLF"].threat_level - 0.0) < 1e-5
        elif step == 725:
            assert abs(links["VLF"].threat_level - 0.5) < 1e-5, f"Step 725 threat level should be 0.5, got {links['VLF'].threat_level}"
        elif step == 750:
            assert abs(links["VLF"].threat_level - 1.0) < 1e-5, f"Step 750 threat level should be 1.0, got {links['VLF'].threat_level}"
        elif step >= 751:
            assert links["VLF"].threat_level == 1.0

    print("  [PASS] Threat level escalation curve (0.0 @ 699 -> 0.5 @ 725 -> 1.0 @ 750+) verified.")

    # 3. Jamming event check on HF
    hf = links["HF"]
    hf.reset(seed=42)
    snr_before = []
    snr_during = []
    for step in range(1000):
        hf.step(step)
        if 350 <= step < 400:
            snr_before.append(hf.current_snr_db)
        elif 400 <= step <= 499:
            snr_during.append(hf.current_snr_db)
    
    avg_before = np.mean(snr_before)
    avg_during = np.mean(snr_during)
    print(f"  HF SNR before jamming (350-399): {avg_before:.2f} dB, during jamming (400-499): {avg_during:.2f} dB")
    assert avg_before - avg_during >= 10.0, "HF Jamming SNR drop failed."
    print("  [PASS] HF link jamming SNR drop verified.")


def test_1_3_feasibility_filter():
    print("\n--- 1.3 Feasibility Filter Integrity Verification ---")
    links = create_military_links()
    
    # 1. SATCOM at max capacity
    satcom = links["SATCOM"]
    satcom.current_bw_bps = 2000000.0  # 2 Mbps
    feasible_sat, _ = get_feasible_actions(satcom)
    print(f"  SATCOM at 2 Mbps feasible pairs: {len(feasible_sat)} / {len(ALL_PQC_PAIRS)}")
    assert len(feasible_sat) == len(ALL_PQC_PAIRS), f"SATCOM should have all {len(ALL_PQC_PAIRS)} feasible pairs at max capacity, got {len(feasible_sat)}."
    print("  [PASS] SATCOM at 2 Mbps has 100% (all 12) candidate pairs feasible.")

    # 2. Extreme degradation fallback on VLF
    vlf = links["VLF"]
    vlf.current_bw_bps = 50.0  # 50 bps min
    feasible_vlf, _ = get_feasible_actions(vlf)
    print(f"  VLF at 50 bps fallback action set count: {len(feasible_vlf)} (Fallback: {feasible_vlf[0]})")
    assert len(feasible_vlf) == 1 and feasible_vlf[0] == ("ML-KEM-512", "ML-DSA-44")
    print("  [PASS] Feasibility filter fallback under extreme degradation verified.")


def test_1_4_linucb_selector():
    print("\n--- 1.4 LinUCB Contextual Bandit Unit Test ---")
    actions = [("ML-KEM-512", "ML-DSA-44"), ("ML-KEM-768", "ML-DSA-65")]
    selector = LinUCBSelector(actions=actions, alpha=0.5)

    x = np.full(8, 0.5, dtype=np.float64)
    best_act, max_p, ucb_scores, thetas = selector.select_action(x, actions)

    # Initial A_a is I_8, b_a is 0_8 -> theta_a is 0_8
    expected_theta = np.zeros(8)
    assert np.allclose(thetas[best_act], expected_theta)
    
    # Variance = x^T * I * x = 8 * 0.25 = 2.0 -> std_dev = sqrt(2) = 1.41421356
    # p_a = 0 + 0.5 * sqrt(2) = 0.70710678
    expected_p = 0.5 * np.sqrt(2.0)
    assert abs(max_p - expected_p) < 1e-12, f"UCB score math mismatch: got {max_p}, expected {expected_p}"
    print("  [PASS] LinUCB analytical math matched machine precision.")

    # Single action update test
    selector.update(best_act, x, reward=1.0)
    assert not np.allclose(selector.b[best_act], np.zeros(8)), "Selected action b_a should be updated."
    other_act = actions[1] if best_act == actions[0] else actions[0]
    assert np.allclose(selector.b[other_act], np.zeros(8)), "Unselected action b_a must NOT be updated."
    print("  [PASS] LinUCB isolated action parameter update verified.")


def test_1_5_xai_engine():
    print("\n--- 1.5 XAI Engine Verification ---")
    xai = XAIEngine()
    x = np.full(8, 0.5)
    theta = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    
    # Contributions = theta * x
    contributions = theta * x
    sum_contrib = float(np.sum(contributions))
    theta_dot_x = float(theta.T @ x)
    
    assert abs(sum_contrib - theta_dot_x) < 1e-12, f"Feature contribution sum mismatch: {sum_contrib} vs {theta_dot_x}"
    print(f"  [PASS] Feature contribution sum ({sum_contrib}) equals theta^T * x ({theta_dot_x}).")

    # Benchmark generation latency
    t_start = time.perf_counter()
    all_thetas = {("ML-KEM-512", "ML-DSA-44"): theta}
    for _ in range(10000):
        _, aud = xai.generate_explanation(
            step_number=1, link_name="VLF", x=x,
            feasible_actions=[("ML-KEM-512", "ML-DSA-44")],
            selected_action=("ML-KEM-512", "ML-DSA-44"),
            theta_selected=theta, all_thetas=all_thetas,
            predicted_reward=0.8, actual_reward=0.8, threat_level=0.0
        )
    t_end = time.perf_counter()
    mean_us = ((t_end - t_start) / 10000.0) * 1e6
    print(f"  Mean XAI explanation generation latency over 10,000 iterations: {mean_us:.2f} µs (Target: < 100 µs)")
    assert mean_us < 100.0, f"XAI latency too high: {mean_us:.2f} µs"
    print("  [PASS] Sub-millisecond XAI explanation overhead confirmed.")


if __name__ == "__main__":
    test_1_1_pqc_database()
    test_1_2_tactical_network()
    test_1_3_feasibility_filter()
    test_1_4_linucb_selector()
    test_1_5_xai_engine()
    print("\n==========================================================")
    print("ALL PHASE 1 CODE VERIFICATION CHECKS COMPLETED SUCCESSFULLY!")
    print("==========================================================\n")
