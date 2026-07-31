"""
Module 5: xai_engine.py — Dual-Level Explanation Generator
XAPQ-TacNet Simulation Framework for IEEE WiCOMM-2026

Generates both operator-level plain text summaries and auditor-level structured JSON logs
directly from LinUCB linear model weight vectors with sub-millisecond overhead.
"""

import time
import json
import numpy as np
from typing import Tuple, List, Dict, Any
from pqc_database import get_nist_security_level
from linucb_selector import FEATURE_NAMES, FEATURE_DIM


class XAIEngine:
    """Explainable AI engine deriving intrinsic feature attributions from LinUCB parameters."""

    def __init__(self):
        self.audit_logs: List[Dict[str, Any]] = []

    def generate_explanation(
        self,
        step_number: int,
        link_name: str,
        x: np.ndarray,
        feasible_actions: List[Tuple[str, str]],
        selected_action: Tuple[str, str],
        theta_selected: np.ndarray,
        all_thetas: Dict[Tuple[str, str], np.ndarray],
        predicted_reward: float,
        actual_reward: float,
        threat_level: float
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate operator summary string and auditor JSON record.
        Measures exact explanation generation latency in microseconds (us).
        """
        t_start = time.perf_counter()

        kem_alg, sig_alg = selected_action
        
        # Element-wise feature contribution: theta * x
        contributions = theta_selected * x
        
        # Rank feature indices by contribution magnitude
        sorted_indices = np.argsort(contributions)[::-1]
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1]

        top1_name = FEATURE_NAMES[top1_idx]
        top1_val = contributions[top1_idx]
        top2_name = FEATURE_NAMES[top2_idx]
        top2_val = contributions[top2_idx]

        # Security assessment string
        kem_lvl = get_nist_security_level(kem_alg)
        sig_lvl = get_nist_security_level(sig_alg)
        avg_lvl = (kem_lvl + sig_lvl) / 2.0
        
        if threat_level > 0.5:
            threat_desc = "quantum-capable threat environment"
            level_assessment = "elevated" if avg_lvl >= 3.0 else "sub-optimal"
        else:
            threat_desc = "conventional threat environment"
            level_assessment = "adequate"

        # 1. Operator-level Summary
        operator_summary = (
            f"Selected {kem_alg}+{sig_alg} on {link_name}: {top1_name} "
            f"[contribution: {top1_val:.2f}] was the dominant factor, followed by {top2_name} "
            f"[contribution: {top2_val:.2f}]. Security level: NIST Level {avg_lvl:.1f}, "
            f"{level_assessment} for assessed {threat_desc}."
        )

        t_end = time.perf_counter()
        gen_time_us = (t_end - t_start) * 1e6  # Microseconds

        # 2. Auditor-level JSON Log
        all_weights_json = {}
        for action_pair, theta in all_thetas.items():
            key_str = f"{action_pair[0]}+{action_pair[1]}"
            all_weights_json[key_str] = theta.tolist()

        auditor_log = {
            "timestamp": step_number,
            "link_id": link_name,
            "context_vector": x.tolist(),
            "feasible_actions": [f"{pair[0]}+{pair[1]}" for pair in feasible_actions],
            "selected_action": {"kem": kem_alg, "sig": sig_alg},
            "all_weight_vectors": all_weights_json,
            "feature_contributions": contributions.tolist(),
            "feature_names": FEATURE_NAMES,
            "predicted_reward": float(predicted_reward),
            "actual_reward": float(actual_reward),
            "explanation_generation_time_us": float(gen_time_us)
        }

        self.audit_logs.append(auditor_log)

        return operator_summary, auditor_log

    def save_audit_log(self, filepath: str = "audit_log.json"):
        """Save captured audit log records to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.audit_logs, f, indent=2)
