"""
Module 10: adversarial_detector.py — Three-Signal Channel-Spoofing Detector with Isolated Anomaly Filtering
XAPQ-TacNet Simulation Framework for IEEE WiCOMM-2026

Implements:
1. Per-Link-Type Anomaly Thresholds (VLF, HF, UHF, SATCOM)
2. Correlated Multi-Feature vs. Isolated Single-Feature Anomaly Filter
3. Temporal Rate-of-Change Filter (Signal 3)
4. Three-Signal Weighted Alert Engine (Score >= 2 -> CRITICAL, Score == 1 + isolated -> WARNING)

Reference: Proakis, Digital Communications, 5th ed., Ch. 5
BPSK and QPSK (Gray coded) have IDENTICAL per-bit BER vs Eb/N0:
Pb = 0.5 * erfc(sqrt(Eb/N0))
"""

import collections
import numpy as np
from scipy.special import erfc
from typing import Dict, Any, List, Optional, Tuple, Union

from linucb_selector import FEATURE_NAMES


class AdversarialDetector:
    """
    Three-Signal Adversarial Channel-Spoofing Detector.
    
    Signal 1: Physical-Layer Consistency Check (BER vs reported Eb/N0).
    Signal 2: Isolated XAI Feature Contribution Anomaly Check (z-score deviation).
    Signal 3: Temporal Rate-of-Change Filter (abrupt vs gradual shift).
    """

    def __init__(
        self,
        window_size: int = 20,
        tolerance_band: float = 10.0,
        anomaly_thresholds: Optional[Union[float, Dict[str, float]]] = None
    ):
        self.window_size = window_size
        self.tolerance_band = tolerance_band
        
        if isinstance(anomaly_thresholds, dict):
            self.anomaly_thresholds = anomaly_thresholds
        elif isinstance(anomaly_thresholds, (float, int)):
            self.anomaly_thresholds = {l_type: float(anomaly_thresholds) for l_type in ['VLF', 'HF', 'UHF', 'SATCOM']}
        else:
            # Default per-link thresholds
            self.anomaly_thresholds = {'VLF': 3.0, 'HF': 3.5, 'UHF': 3.5, 'SATCOM': 4.5}

        self.contribution_history = {}
        self.context_history = {}
        self.previous_security_level = {}
        self.clean_baselines = {}
        self.feature_names = FEATURE_NAMES

        # BPSK and QPSK (Gray coded) have IDENTICAL per-bit BER vs Eb/N0
        self.channel_models = {
            'VLF':    lambda ebno_linear: 0.5 * erfc(np.sqrt(ebno_linear)),  # BPSK
            'HF':     lambda ebno_linear: 0.5 * erfc(np.sqrt(ebno_linear)),  # BPSK
            'UHF':    lambda ebno_linear: 0.5 * erfc(np.sqrt(ebno_linear)),  # QPSK Gray
            'SATCOM': lambda ebno_linear: 0.5 * erfc(np.sqrt(ebno_linear)),  # QPSK Gray
        }

    def check_physical_consistency(
        self,
        link_type: str,
        reported_ebno_db: float,
        observed_ber: float
    ) -> Dict[str, Any]:
        """Signal 1: Detect if observed BER is consistent with reported Eb/N0 for link modulation."""
        ebno_linear = 10.0 ** (reported_ebno_db / 10.0)
        raw_ber = float(self.channel_models[link_type](ebno_linear))
        expected_ber = float(np.clip(raw_ber, 1e-8, 0.5))
        
        ratio = float(observed_ber / (expected_ber + 1e-15))
        consistent = (1.0 / self.tolerance_band) < ratio < self.tolerance_band
        
        return {
            'consistent': consistent,
            'expected_ber': expected_ber,
            'observed_ber': observed_ber,
            'ratio': ratio
        }

    def check_contribution_anomaly(
        self,
        link_id: str,
        contributions: np.ndarray,
        is_downgrade: bool
    ) -> Dict[str, Any]:
        """
        Signal 2: Detect if feature contributions are anomalous during a security downgrade.
        Filters out correlated multi-feature degradation (>= 3 features) to isolate spoofing (<= 2 features).
        """
        if link_id not in self.contribution_history:
            self.contribution_history[link_id] = collections.deque(maxlen=self.window_size)

        history = self.contribution_history[link_id]
        
        if len(history) < self.window_size // 2:
            history.append(np.array(contributions, copy=True))
            self.clean_baselines[link_id] = (np.mean(list(history), axis=0), np.std(list(history), axis=0) + 1e-8)
            return {
                'anomalous': False,
                'max_deviation': 0.0,
                'anomalous_feature_idx': 0,
                'anomalous_feature_name': self.feature_names[0],
                'significant_feature_count': 0,
                'is_isolated': False,
                'reason': 'insufficient_history'
            }
        
        # Lock in clean baseline once window is full
        if link_id in self.clean_baselines:
            mean_contrib, std_contrib = self.clean_baselines[link_id]
        else:
            historical = np.array(list(history))
            mean_contrib = np.mean(historical, axis=0)
            std_contrib = np.std(historical, axis=0) + 1e-8
            if len(history) >= self.window_size:
                self.clean_baselines[link_id] = (mean_contrib, std_contrib)
        
        # Compute per-feature z-score deviations
        deviations = np.abs(contributions - mean_contrib) / std_contrib
        max_deviation = float(np.max(deviations))
        anomalous_feature_idx = int(np.argmax(deviations))
        
        # Count features deviating significantly (> 2.0 sigma)
        significant_features = int(np.sum(deviations > 2.0))
        
        # Isolated anomaly check: spoofing manipulates 1-2 features in isolation
        is_isolated = (significant_features <= 2)
        
        link_threshold = self.anomaly_thresholds.get(link_id, 3.5)
        is_large = (max_deviation > link_threshold)
        
        # Only flag if large, isolated, AND during a security downgrade
        is_anomalous = bool(is_large and is_isolated and is_downgrade)
        
        history.append(np.array(contributions, copy=True))
        
        return {
            'anomalous': is_anomalous,
            'max_deviation': max_deviation,
            'anomalous_feature_idx': anomalous_feature_idx,
            'anomalous_feature_name': self.feature_names[anomalous_feature_idx],
            'significant_feature_count': significant_features,
            'is_isolated': is_isolated,
            'all_deviations': deviations.tolist()
        }

    def check_rate_of_change(
        self,
        link_id: str,
        feature_idx: int,
        current_context: np.ndarray
    ) -> Dict[str, Any]:
        """Signal 3: Temporal rate-of-change check for abrupt vs. gradual feature shifts."""
        if link_id not in self.context_history:
            self.context_history[link_id] = collections.deque(maxlen=5)

        history = self.context_history[link_id]
        
        if len(history) < 3:
            history.append(np.array(current_context, copy=True))
            return {'is_abrupt': False, 'rate_ratio': 1.0}

        recent = np.array(list(history))
        recent_diffs = np.abs(np.diff(recent[:, feature_idx]))
        mean_past_rate = float(np.mean(recent_diffs) + 1e-8)
        
        current_rate = float(abs(current_context[feature_idx] - recent[-1, feature_idx]))
        rate_ratio = float(current_rate / mean_past_rate)
        
        history.append(np.array(current_context, copy=True))
        
        is_abrupt = bool(rate_ratio > 3.0)
        return {'is_abrupt': is_abrupt, 'rate_ratio': rate_ratio}

    def evaluate(
        self,
        link_id: str,
        link_type: str,
        context: np.ndarray,
        contributions: np.ndarray,
        current_security_level: int,
        reported_ebno_db: float,
        observed_ber: float
    ) -> Dict[str, Any]:
        """
        Three-Signal Scoring Evaluation Engine.
        Score = Signal1 (Physical) + Signal2 (Isolated XAI Anomaly) + Signal3 (Abrupt Rate of Change)
        """
        prev_level = self.previous_security_level.get(link_id, current_security_level)
        is_downgrade = current_security_level < prev_level
        self.previous_security_level[link_id] = current_security_level

        # Evaluate 3 Signals
        sig1_res = self.check_physical_consistency(link_type, reported_ebno_db, observed_ber)
        sig2_res = self.check_contribution_anomaly(link_id, contributions, is_downgrade)
        sig3_res = self.check_rate_of_change(link_id, sig2_res['anomalous_feature_idx'], context)

        signal1_fired = 1 if not sig1_res['consistent'] else 0
        signal2_fired = 1 if sig2_res['anomalous'] else 0
        signal3_fired = 1 if sig3_res['is_abrupt'] else 0

        score = signal1_fired + signal2_fired + signal3_fired
        is_isolated = sig2_res.get('is_isolated', False)

        if score >= 2:
            alert = {
                'alert_level': 'CRITICAL',
                'type': 'POTENTIAL_CHANNEL_SPOOFING',
                'signals_fired': score,
                'physical_evidence': f"BER/Eb_N0 ratio {sig1_res['ratio']:.1f}x outside limits",
                'behavioral_evidence': f"{sig2_res['anomalous_feature_name']} dev: {sig2_res['max_deviation']:.1f} sigma (isolated)",
                'recommendation': 'MAINTAIN previous security level',
                'action': 'BLOCK_DOWNGRADE',
                'previous_security_level': prev_level
            }
        elif score == 1 and is_isolated and is_downgrade:
            alert = {
                'alert_level': 'WARNING',
                'type': 'SUSPICIOUS_DEGRADATION',
                'signals_fired': score,
                'evidence': 'Single isolated signal anomaly',
                'action': 'FLAG_FOR_REVIEW',
                'previous_security_level': prev_level
            }
        else:
            alert = {
                'alert_level': 'CLEAR',
                'signals_fired': score,
                'action': 'ALLOW',
                'is_downgrade': is_downgrade
            }

        alert['signal1_raw'] = sig1_res
        alert['signal2_raw'] = sig2_res
        alert['signal3_raw'] = sig3_res
        alert['is_downgrade'] = is_downgrade

        return alert


def verify_ber_model():
    """Verification test for per-bit BER model across links."""
    detector = AdversarialDetector()
    test_points = {0: 7.865e-02, 8: 1.909e-04, 10: 3.872e-06}
    
    for ebno_db, expected in test_points.items():
        for link_type in ['VLF', 'HF', 'UHF', 'SATCOM']:
            result = detector.check_physical_consistency(link_type, ebno_db, expected)
            ber = result['expected_ber']
            assert abs(ber - expected) / expected < 0.001
            assert 0.99 < result['ratio'] < 1.01
    
    print("BER model verification PASSED — all links consistent")


if __name__ == '__main__':
    verify_ber_model()
