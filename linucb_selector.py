"""
Module 4: linucb_selector.py — Per-Link LinUCB Contextual Bandit Selector with Warm-Starting
XAPQ-TacNet Simulation Framework for IEEE WiCOMM-2026

Implements:
1. Per-Link-Type Disjoint LinUCB Agents (VLF, HF, UHF, SATCOM)
2. Decaying Alpha Schedule for Dynamic Exploration-Exploitation
3. 10-Dimensional Context Vector with Threat-Security Interaction Terms
4. Rule-Based Policy Warm-Starting (eliminates cold-start period)
5. Optimistic Security Initialization Bias
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from pqc_database import get_nist_security_level, total_handshake_time, get_total_payload_bytes
from tactical_network import TacticalLink

# Expanded Feature dimension d = 10
FEATURE_DIM = 10
FEATURE_NAMES = [
    "bandwidth",
    "latency_budget",
    "snr",
    "ber",
    "compute_capability",
    "energy_remaining",
    "mission_criticality",
    "threat_level",
    "security_headroom",
    "threat_security_interaction"
]


def extract_context_vector(link: TacticalLink, feasible_actions: List[Tuple[str, str]] = None) -> np.ndarray:
    """
    Construct d=10 feature context vector x for the given tactical link state.
    Features 1-8: Base normalized operational & environmental features
    Feature 9: Security headroom on current link
    Feature 10: Threat-security gap interaction term
    """
    # 1. Normalized Bandwidth
    norm_bw = (link.current_bw_bps - link.bw_min_bps) / max(1.0, (link.bw_max_bps - link.bw_min_bps))
    norm_bw = float(np.clip(norm_bw, 0.0, 1.0))

    # 2. Normalized Latency Budget
    norm_lat_budget = float(np.clip(link.latency_budget_sec / 30.0, 0.0, 1.0))

    # 3. Normalized SNR
    norm_snr = (link.current_snr_db - link.snr_min_db) / max(1.0, (link.snr_max_db - link.snr_min_db))
    norm_snr = float(np.clip(norm_snr, 0.0, 1.0))

    # 4. Normalized BER
    max_ber = 0.5
    norm_ber = float(np.clip(1.0 - (link.current_ber / max_ber), 0.0, 1.0))

    # 5. Compute Capability
    max_clock_mhz = 1800.0
    norm_compute = float(np.log(link.clock_speed_mhz) / np.log(max_clock_mhz))

    # 6. Energy Remaining
    norm_energy = float(np.clip(link.energy_remaining, 0.0, 1.0))

    # 7. Mission Criticality
    norm_crit = float(link.mission_criticality)

    # 8. Threat Level
    norm_threat = float(link.threat_level)

    # 9. Security Headroom Feature
    if feasible_actions and len(feasible_actions) > 0:
        sec_levels = [get_nist_security_level(a[0]) for a in feasible_actions]
        sec_headroom = float((max(sec_levels) - min(sec_levels)) / 5.0)
    else:
        sec_headroom = 0.0

    # 10. Threat-Security Gap Interaction
    threat_sec_gap = float(norm_threat * sec_headroom)

    x = np.array([
        norm_bw,
        norm_lat_budget,
        norm_snr,
        norm_ber,
        norm_compute,
        norm_energy,
        norm_crit,
        norm_threat,
        sec_headroom,
        threat_sec_gap
    ], dtype=np.float64)

    return x


def compute_reward(
    action: Tuple[str, str],
    link: TacticalLink,
    total_time: float
) -> Tuple[float, float, float, float]:
    """Compute multi-objective reward score for selected (KEM, SIG) pair."""
    kem_alg, sig_alg = action
    
    # 1. Security Score
    kem_lvl = get_nist_security_level(kem_alg)
    sig_lvl = get_nist_security_level(sig_alg)
    security_score = ((kem_lvl + sig_lvl) / 2.0) / 5.0

    # 2. Latency Ratio
    latency_ratio = min(1.0, total_time / link.latency_budget_sec)

    # 3. Bandwidth Ratio
    total_bytes = get_total_payload_bytes(kem_alg, sig_alg)
    max_capacity_bytes = (link.current_bw_bps * link.latency_budget_sec) / 8.0
    bandwidth_ratio = min(1.0, total_bytes / max(1.0, max_capacity_bytes))

    # 4. Dynamic Weighting based on Threat Level
    if link.threat_level < 0.5:
        l1, l2, l3 = 0.30, 0.40, 0.30
    else:
        l1, l2, l3 = 0.60, 0.25, 0.15

    reward = l1 * security_score + l2 * (1.0 - latency_ratio) + l3 * (1.0 - bandwidth_ratio)
    return float(reward), security_score, latency_ratio, bandwidth_ratio


class LinUCBAgent:
    """Disjoint LinUCB agent for a single link type."""

    def __init__(self, actions: List[Tuple[str, str]], d: int = FEATURE_DIM):
        self.actions = actions
        self.d = d
        self.A = {action: np.eye(d, dtype=np.float64) for action in actions}
        self.b = {action: np.zeros(d, dtype=np.float64) for action in actions}

    def select_action(
        self,
        x: np.ndarray,
        feasible_actions: List[Tuple[str, str]],
        alpha: float
    ) -> Tuple[Tuple[str, str], float, Dict[Tuple[str, str], float], Dict[Tuple[str, str], np.ndarray]]:
        best_action = None
        max_p = -float("inf")
        ucb_scores = {}
        theta_dict = {}

        for action in feasible_actions:
            if action not in self.A:
                self.A[action] = np.eye(self.d, dtype=np.float64)
                self.b[action] = np.zeros(self.d, dtype=np.float64)

            A_inv = np.linalg.inv(self.A[action])
            theta_a = A_inv @ self.b[action]
            theta_dict[action] = theta_a

            variance = float(x.T @ A_inv @ x)
            std_dev = np.sqrt(max(0.0, variance))
            p_a = float(theta_a.T @ x + alpha * std_dev)
            
            ucb_scores[action] = p_a

            if p_a > max_p:
                max_p = p_a
                best_action = action

        return best_action, max_p, ucb_scores, theta_dict

    def update(self, action: Tuple[str, str], x: np.ndarray, reward: float):
        if action not in self.A:
            self.A[action] = np.eye(self.d, dtype=np.float64)
            self.b[action] = np.zeros(self.d, dtype=np.float64)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x


class PerLinkLinUCB:
    """
    Per-Link-Type LinUCB Selector managing dedicated agents per link type 
    with decaying alpha schedules, warm-starting, and optimistic security biases.
    """

    def __init__(
        self,
        actions: List[Tuple[str, str]],
        link_types: List[str] = None,
        d: int = FEATURE_DIM,
        alpha_start: float = 1.0,
        alpha_min: float = 0.05,
        decay_rate: float = 0.005
    ):
        if link_types is None:
            link_types = ["VLF", "HF", "UHF", "SATCOM"]
        self.actions = actions
        self.link_types = link_types
        self.d = d
        self.alpha_start = alpha_start
        self.alpha_min = alpha_min
        self.decay_rate = decay_rate

        self.agents = {l_type: LinUCBAgent(actions=actions, d=d) for l_type in link_types}
        self.step_counts = {l_type: 0 for l_type in link_types}

    def get_alpha(self, link_type: str) -> float:
        step = self.step_counts.get(link_type, 0)
        return float(max(self.alpha_min, self.alpha_start * np.exp(-self.decay_rate * step)))

    def select_action(
        self,
        link_type: str,
        x: np.ndarray,
        feasible_actions: List[Tuple[str, str]]
    ) -> Tuple[Tuple[str, str], float, Dict[Tuple[str, str], float], Dict[Tuple[str, str], np.ndarray]]:
        alpha = self.get_alpha(link_type)
        agent = self.agents[link_type]
        return agent.select_action(x, feasible_actions, alpha=alpha)

    def update(self, link_type: str, action: Tuple[str, str], x: np.ndarray, reward: float):
        agent = self.agents[link_type]
        agent.update(action, x, reward)
        self.step_counts[link_type] += 1

    def warm_start(self, link_type: str, link_obj: TacticalLink, n_warmup: int = 50, seed: int = 42):
        """Pre-populate LinUCB weights using rule-based policy synthetic experience."""
        from baselines import RuleBasedSelector
        from feasibility_filter import get_feasible_actions
        
        rule_based = RuleBasedSelector()
        agent = self.agents[link_type]
        rng = np.random.RandomState(seed)

        for i in range(n_warmup):
            # Create synthetic channel state
            link_obj.current_bw_bps = rng.uniform(link_obj.bw_min_bps, link_obj.bw_max_bps)
            link_obj.current_snr_db = rng.uniform(link_obj.snr_min_db, link_obj.snr_max_db)
            link_obj.threat_level = rng.choice([0.2, 0.5, 0.8])
            
            feasible_actions, _ = get_feasible_actions(link_obj)
            if not feasible_actions:
                continue

            x = extract_context_vector(link_obj, feasible_actions)
            selected_action = rule_based.select_action(link_obj, feasible_actions)
            
            t_handshake, _, _ = total_handshake_time(
                selected_action[0], selected_action[1], link_obj.clock_speed_mhz, link_obj.current_bw_bps
            )
            reward, _, _, _ = compute_reward(selected_action, link_obj, t_handshake)
            
            agent.update(selected_action, x, reward)

        # Reset step count so alpha decay starts fresh for online simulation
        self.step_counts[link_type] = 0

    def apply_security_bias(self, link_type: str, bias_strength: float = 0.1):
        """Add optimistic bias toward higher-security actions to prevent premature abandonment."""
        agent = self.agents[link_type]
        for action in self.actions:
            sec_lvl = get_nist_security_level(action[0])
            bias = (sec_lvl / 5.0) * bias_strength
            agent.b[action] += bias * np.ones(self.d, dtype=np.float64)
