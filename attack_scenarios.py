"""
Module 11: attack_scenarios.py — Adversarial Channel Attack Simulator
XAPQ-TacNet Simulation Framework for IEEE WiCOMM-2026

Defines adversarial channel manipulation scenarios overlaying the tactical network simulator.
Produces reported_state (what the selector sees) vs. actual_state (physical reality).
"""

from typing import Dict, Any, Tuple


class AttackScenarioManager:
    """
    Overlays adversarial channel manipulations on top of normal link dynamics.
    
    Scenario A: Sudden SATCOM Eb/N0 Spoofing (steps 800–850).
    Scenario B: Gradual UHF Bandwidth Manipulation (steps 850–950).
    """

    def __init__(self):
        self.scenario_a = {
            'link': 'SATCOM',
            'start_step': 800,
            'end_step': 850,
            'type': 'ebno_spoof',
            'magnitude_db': 12.0  # Eb/N0 reduction in dB
        }
        self.scenario_b = {
            'link': 'SATCOM',
            'start_step': 850,
            'end_step': 950,
            'type': 'bandwidth_manipulation',
            'max_reduction_fraction': 0.03  # 3% per step at max
        }

    def apply(self, step: int, link_type: str, actual_state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """
        Applies attack scenarios if applicable to current step and link.
        
        Returns: (reported_state, actual_state, is_under_attack)
        """
        reported_state = actual_state.copy()
        is_under_attack = False

        # Scenario A: Sudden SATCOM Eb/N0 Spoofing (steps 800 - 850)
        if (
            link_type == self.scenario_a['link'] and
            self.scenario_a['start_step'] <= step <= self.scenario_a['end_step']
        ):
            reported_state['ebno_db'] = actual_state['ebno_db'] - self.scenario_a['magnitude_db']
            # BER stays at ACTUAL value (physical channel unchanged)
            # Signal 1 detects BER vs Eb/N0 mismatch
            reported_state['ber'] = actual_state['ber']
            is_under_attack = True

        # Scenario B: Gradual UHF Bandwidth Manipulation (steps 850 - 950)
        if (
            link_type == self.scenario_b['link'] and
            self.scenario_b['start_step'] <= step <= self.scenario_b['end_step']
        ):
            steps_into_attack = step - self.scenario_b['start_step']
            reduction = min(0.95, 0.03 * steps_into_attack)  # 3% per step up to 95%
            reported_state['bandwidth_bps'] = actual_state['bandwidth_bps'] * (1.0 - reduction)
            # Eb/N0 and BER unchanged — physical channel fine
            # Signal 2 detects feature contribution anomaly
            is_under_attack = True

        return reported_state, actual_state, is_under_attack
