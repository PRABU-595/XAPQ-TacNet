"""
Module 6: baselines.py — Four Baseline PQC Selection Algorithms
XAPQ-TacNet Simulation Framework for IEEE WiCOMM-2026

Implements the four benchmark comparison strategies:
1. Static Policy: Always requests highest security (ML-KEM-1024 + ML-DSA-65)
2. Random Policy: Uniform random selection among feasible pairs
3. Rule-Based Heuristic: Hardcoded bandwidth threshold decision tree
4. Exhaustive Scalarized Oracle: Brute-force optimal selection.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from pqc_database import get_nist_security_level, total_handshake_time, get_total_payload_bytes
from tactical_network import TacticalLink

# ------------------------------------------------------------------------------
# Baseline 1: Static Selection Policy
# ------------------------------------------------------------------------------
class StaticSelector:
    """Always selects ML-KEM-1024 + ML-DSA-65. Fallback if infeasible."""
    
    def select_action(self, feasible_actions: List[Tuple[str, str]]) -> Tuple[str, str]:
        target = ("ML-KEM-1024", "ML-DSA-65")
        if target in feasible_actions:
            return target
        fallback = ("ML-KEM-512", "ML-DSA-44")
        if fallback in feasible_actions:
            return fallback
        return feasible_actions[0]


# ------------------------------------------------------------------------------
# Baseline 2: Uniform Random Selection Policy
# ------------------------------------------------------------------------------
class RandomSelector:
    """Selects uniformly at random from feasible actions."""

    def select_action(self, feasible_actions: List[Tuple[str, str]]) -> Tuple[str, str]:
        idx = np.random.choice(len(feasible_actions))
        return feasible_actions[idx]


# ------------------------------------------------------------------------------
# Baseline 3: Rule-Based Heuristic Selector
# ------------------------------------------------------------------------------
class RuleBasedSelector:
    """Deterministic rule-based decision tree based on link bandwidth thresholds."""

    def select_action(self, link: TacticalLink, feasible_actions: List[Tuple[str, str]]) -> Tuple[str, str]:
        bw = link.current_bw_bps
        if bw < 1000.0:
            target = ("ML-KEM-512", "ML-DSA-44")
        elif bw < 50000.0:
            target = ("ML-KEM-768", "ML-DSA-44")
        elif bw < 500000.0:
            target = ("ML-KEM-768", "ML-DSA-65")
        else:
            target = ("ML-KEM-1024", "ML-DSA-65")

        if target in feasible_actions:
            return target
        
        # Fallback to closest feasible
        return feasible_actions[0]


# ------------------------------------------------------------------------------
# Baseline 4: Exhaustive Scalarized Oracle
# ------------------------------------------------------------------------------
class ExhaustiveOracleSelector:
    """
    Exhaustive Scalarized Oracle.
    Brute-force searches the true optimum of the reward function over the feasible action space.
    """

    def select_action(
        self,
        link: TacticalLink,
        feasible_actions: List[Tuple[str, str]]
    ) -> Tuple[str, str]:
        if len(feasible_actions) == 1:
            return feasible_actions[0]

        if link.threat_level < 0.5:
            l1, l2, l3 = 0.30, 0.40, 0.30
        else:
            l1, l2, l3 = 0.60, 0.25, 0.15

        best_idx = 0
        best_val = float("-inf")
        
        for idx, candidate in enumerate(feasible_actions):
            total_time, _, _ = total_handshake_time(
                candidate[0], candidate[1], link.clock_speed_mhz, link.current_bw_bps
            )
            lat_ratio = min(1.0, total_time / link.latency_budget_sec)
            total_bytes = get_total_payload_bytes(candidate[0], candidate[1])
            max_cap_bytes = (link.current_bw_bps * link.latency_budget_sec) / 8.0
            bw_ratio = min(1.0, total_bytes / max(1.0, max_cap_bytes))
            sec_score = ((get_nist_security_level(candidate[0]) + get_nist_security_level(candidate[1])) / 2.0) / 5.0

            score = l1 * sec_score + l2 * (1.0 - lat_ratio) + l3 * (1.0 - bw_ratio)
            if score > best_val:
                best_val = score
                best_idx = idx

        return feasible_actions[best_idx]
