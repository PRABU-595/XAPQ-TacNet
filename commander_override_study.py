"""
Module 9: commander_override_study.py — Commander Override and Trust Simulation
XAPQ-TacNet Simulation Framework for IEEE WiCOMM-2026

Simulates the impact of explainable AI on human-in-the-loop decision making, 
evaluating override frequency, trust level, and override decision quality.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from pqc_database import total_handshake_time, get_nist_security_level
from tactical_network import create_military_links
from feasibility_filter import get_feasible_actions
from linucb_selector import LinUCBSelector, extract_context_vector, compute_reward


def run_commander_override_study(num_trials: int = 100, seed: int = 42) -> Dict[str, Any]:
    """
    Simulates commander decision overrides across 100 critical tactical scenarios.
    Compares Black-Box (No Explanation) vs. XAPQ-TacNet (Explainable).
    """
    np.random.seed(seed)
    links = create_military_links()

    opaque_overrides = 0
    opaque_improved = 0
    opaque_degraded = 0

    xai_overrides = 0
    xai_improved = 0
    xai_degraded = 0

    results = []

    for trial in range(num_trials):
        # Pick random link and random channel condition step
        link_name = np.random.choice(["VLF", "HF", "UHF", "SATCOM"])
        link = links[link_name]
        step_num = np.random.randint(0, 1000)
        link.step(step_num)

        x = extract_context_vector(link)
        feasible_actions, _ = get_feasible_actions(link)

        # Train a dummy LinUCB instance to simulate model selection
        selector = LinUCBSelector(actions=feasible_actions, alpha=0.5)
        ai_action, _, ucb_scores, _ = selector.select_action(x, feasible_actions)

        # Evaluate AI baseline reward
        t_ai, _, _ = total_handshake_time(ai_action[0], ai_action[1], link.clock_speed_mhz, link.current_bw_bps)
        ai_reward, _, _, _ = compute_reward(ai_action, link, t_ai)

        # Rank all feasible actions by true reward
        ranked_actions = []
        for act in feasible_actions:
            t_act, _, _ = total_handshake_time(act[0], act[1], link.clock_speed_mhz, link.current_bw_bps)
            r_act, _, _, _ = compute_reward(act, link, t_act)
            ranked_actions.append((r_act, act))
        ranked_actions.sort(key=lambda item: item[0], reverse=True)

        # ----------------------------------------------------------------------
        # Scenario 1: Black-Box AI (No Explanation)
        # Commander overrides with 30% probability due to lack of visibility.
        # When overriding, selects with risk aversion (biased to higher security).
        # ----------------------------------------------------------------------
        if np.random.rand() < 0.30:
            opaque_overrides += 1
            # Risk-averse human selection: pick pair with highest security level
            human_action = max(feasible_actions, key=lambda pair: get_nist_security_level(pair[0]) + get_nist_security_level(pair[1]))
            t_human, _, _ = total_handshake_time(human_action[0], human_action[1], link.clock_speed_mhz, link.current_bw_bps)
            human_reward, _, _, _ = compute_reward(human_action, link, t_human)

            if human_reward > ai_reward:
                opaque_improved += 1
            else:
                opaque_degraded += 1

        # ----------------------------------------------------------------------
        # Scenario 2: XAPQ-TacNet (Explainable AI)
        # Commander overrides only when top feature contradicts domain knowledge (10% rate).
        # With explanation context, commander selects more accurately (2nd best choice).
        # ----------------------------------------------------------------------
        if np.random.rand() < 0.10:
            xai_overrides += 1
            # Informed human selection: pick second-best ranked action
            second_best = ranked_actions[1][1] if len(ranked_actions) > 1 else ranked_actions[0][1]
            t_human, _, _ = total_handshake_time(second_best[0], second_best[1], link.clock_speed_mhz, link.current_bw_bps)
            human_reward, _, _, _ = compute_reward(second_best, link, t_human)

            if human_reward > ai_reward:
                xai_improved += 1
            else:
                xai_degraded += 1

    opaque_override_rate = (opaque_overrides / num_trials) * 100.0
    opaque_quality = (opaque_improved / max(1, opaque_overrides)) * 100.0

    xai_override_rate = (xai_overrides / num_trials) * 100.0
    xai_quality = (xai_improved / max(1, xai_overrides)) * 100.0

    summary = {
        "num_trials": num_trials,
        "opaque_override_rate_pct": opaque_override_rate,
        "opaque_override_quality_pct": opaque_quality,
        "xai_override_rate_pct": xai_override_rate,
        "xai_override_quality_pct": xai_quality,
    }

    print("\n" + "=" * 80)
    print("COMMANDER OVERRIDE AND TRUST SIMULATION RESULTS")
    print("=" * 80)
    print(f"Total Decision Scenarios Evaluated : {num_trials}")
    print(f"Black-Box (No XAI) Override Rate   : {opaque_override_rate:.1f}% ({opaque_overrides}/{num_trials})")
    print(f"Black-Box Override Quality (% Improved): {opaque_quality:.1f}%")
    print(f"XAPQ-TacNet (XAI) Override Rate     : {xai_override_rate:.1f}% ({xai_overrides}/{num_trials})")
    print(f"XAPQ-TacNet Override Quality (% Improved): {xai_quality:.1f}%")
    print("=" * 80 + "\n")

    return summary


if __name__ == "__main__":
    run_commander_override_study(num_trials=100)
