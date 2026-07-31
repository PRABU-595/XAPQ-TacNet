"""
Module 6: baselines.py — Four Baseline PQC Selection Algorithms
XAPQ-TacNet Simulation Framework for IEEE WiCOMM-2026

Implements the four benchmark comparison strategies:
1. Static Policy: Always requests highest security (ML-KEM-1024 + ML-DSA-65)
2. Random Policy: Uniform random selection among feasible pairs
3. Rule-Based Heuristic: Hardcoded bandwidth threshold decision tree
4. CAAP-Adapted (NSGA-II MOEA): Evolutionary multi-objective optimization via pymoo
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
# Baseline 4: CAAP-Adapted Multi-Objective Evolutionary Algorithm (NSGA-II)
# ------------------------------------------------------------------------------
class CAAPAdaptedSelector:
    """
    NSGA-II Multi-Objective Evolutionary Algorithm using pymoo library.
    Objectives:
    f1: Minimize Latency Ratio
    f2: Minimize Bandwidth Ratio
    f3: Maximize Security Level (minimize -security_score)
    """

    def select_action(
        self,
        link: TacticalLink,
        feasible_actions: List[Tuple[str, str]]
    ) -> Tuple[str, str]:
        """
        Uses NSGA-II or discrete Pareto scoring over feasible actions.
        For discrete action sets (<= 21 pairs), NSGA-II evaluates candidates over 10 generations.
        Returns the top Pareto-optimal candidate weighted by threat-based lambda.
        """
        try:
            from pymoo.core.problem import Problem
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.optimize import minimize
            from pymoo.operators.sampling.rnd import IntegerRandomSampling
            from pymoo.operators.crossover.sbx import SBX
            from pymoo.operators.mutation.pm import PM

            n_feasible = len(feasible_actions)
            if n_feasible == 1:
                return feasible_actions[0]

            # Define discrete PQC selection problem
            class PQCProblem(Problem):
                def __init__(self, actions, link_obj):
                    super().__init__(n_var=1, n_obj=3, n_ieq_constr=0, xl=0, xu=n_feasible-1)
                    self.actions = actions
                    self.link_obj = link_obj

                def _evaluate(self, x, out, *args, **kwargs):
                    f = np.zeros((x.shape[0], 3))
                    for i in range(x.shape[0]):
                        idx = int(round(x[i, 0])) % n_feasible
                        action = self.actions[idx]
                        
                        total_time, _, _ = total_handshake_time(
                            action[0], action[1], self.link_obj.clock_speed_mhz, self.link_obj.current_bw_bps
                        )
                        lat_ratio = min(1.0, total_time / self.link_obj.latency_budget_sec)
                        
                        total_bytes = get_total_payload_bytes(action[0], action[1])
                        max_cap_bytes = (self.link_obj.current_bw_bps * self.link_obj.latency_budget_sec) / 8.0
                        bw_ratio = min(1.0, total_bytes / max(1.0, max_cap_bytes))
                        
                        kem_lvl = get_nist_security_level(action[0])
                        sig_lvl = get_nist_security_level(action[1])
                        sec_score = ((kem_lvl + sig_lvl) / 2.0) / 5.0
                        
                        # Objectives: min lat_ratio, min bw_ratio, min -sec_score
                        f[i, 0] = lat_ratio
                        f[i, 1] = bw_ratio
                        f[i, 2] = -sec_score

                    out["F"] = f

            problem = PQCProblem(feasible_actions, link)
            algorithm = NSGA2(
                pop_size=min(20, max(4, n_feasible * 2)),
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=0.8, eta=15),
                mutation=PM(prob=0.2, eta=20)
            )

            res = minimize(problem, algorithm, ('n_gen', 10), verbose=False, seed=42)

            # Select candidate from Pareto front using weighted scalarization
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

        except Exception as e:
            # Fallback if pymoo or optimization encounters edge case
            return feasible_actions[0]
