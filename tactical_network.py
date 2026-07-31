"""
Module 2: tactical_network.py — Realistic Military Link Simulator
XAPQ-TacNet Simulation Framework for IEEE WiCOMM-2026

Simulates military communication links adhering to MIL-STD specifications:
1. VLF  (MIL-STD-188-141 class): Submarine / deep-bunker communications
2. HF   (MIL-STD-188-110D class): Ground tactical, NVIS waveform
3. UHF  (MIL-STD-188-181 class): Vehicle SDR, squad tactical radio
4. SATCOM (MIL-STD-188-165 class): Ku-band command post uplink / UAV datalink
"""

import numpy as np
from typing import Dict, Any, Tuple


class TacticalLink:
    """Represents a single tactical military communications channel."""
    
    def __init__(
        self,
        name: str,
        mil_std: str,
        bw_range_bps: Tuple[float, float],
        snr_range_db: Tuple[float, float],
        latency_budget_sec: float,
        clock_speed_mhz: float,
        use_case: str
    ):
        self.name = name
        self.mil_std = mil_std
        self.bw_min_bps, self.bw_max_bps = bw_range_bps
        self.snr_min_db, self.snr_max_db = snr_range_db
        self.latency_budget_sec = latency_budget_sec
        self.clock_speed_mhz = clock_speed_mhz
        self.use_case = use_case
        
        # State variables
        self.current_step = 0
        self.current_bw_bps = (self.bw_min_bps + self.bw_max_bps) / 2.0
        self.current_snr_db = (self.snr_min_db + self.snr_max_db) / 2.0
        self.current_ber = self._calculate_ber(self.current_snr_db)
        self.mission_criticality = 0.33  # 0.33 (routine), 0.67 (priority), 1.0 (flash)
        self.threat_level = 0.0         # 0.0 (conventional) to 1.0 (quantum-capable)
        self.energy_remaining = 1.0     # 1.0 down to 0.0 (decays 0.001/step)

    @property
    def current_ebno_db(self) -> float:
        """Alias for current_snr_db following Eb/N0 convention."""
        return self.current_snr_db

    def _calculate_ber(self, snr_db: float) -> float:
        """
        SNR CONVENTION: All SNR values in this simulation represent Eb/N0 
        (energy per bit to noise spectral density ratio) in dB.
        This is the standard metric in MIL-STD link budget calculations.
        
        Important: BPSK and QPSK (Gray coded) have IDENTICAL per-bit BER 
        vs Eb/N0 (Proakis, Digital Communications, 5th ed., Ch. 5).
        This means the adversarial detector's physical consistency check 
        uses the same BER formula for all four link types.
        """
        snr_linear = 10.0 ** (snr_db / 10.0)
        from scipy.special import erfc
        ber = 0.5 * erfc(np.sqrt(snr_linear))
        return float(np.clip(ber, 1e-8, 0.5))

    def reset(self, seed: int = None):
        """Reset link state for a new experiment seed."""
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.current_bw_bps = np.random.uniform(self.bw_min_bps, self.bw_max_bps)
        self.current_snr_db = np.random.uniform(self.snr_min_db, self.snr_max_db)
        self.current_ber = self._calculate_ber(self.current_snr_db)
        self.mission_criticality = 0.33
        self.threat_level = 0.0
        self.energy_remaining = 1.0

    def step(self, step_num: int):
        """Advance link conditions by one time step (0..999)."""
        self.current_step = step_num

        # 1. Energy decay
        self.energy_remaining = max(0.0, 1.0 - 0.001 * step_num)

        # 2. Threat Level Escalation (Steps 700 - 999)
        if step_num < 700:
            self.threat_level = 0.0
        elif 700 <= step_num <= 750:
            self.threat_level = (step_num - 700) / 50.0
        else:
            self.threat_level = 1.0

        # 3. Mission Criticality Sampling
        # After step 700, probability of "flash" mission increases
        if step_num >= 700:
            p_flash = 0.5
            p_priority = 0.35
            p_routine = 0.15
        else:
            p_flash = 0.15
            p_priority = 0.35
            p_routine = 0.50
        
        crit_choice = np.random.choice([0.33, 0.67, 1.0], p=[p_routine, p_priority, p_flash])
        self.mission_criticality = float(crit_choice)

        # 4. Physical Channel Dynamics (SNR & Bandwidth)
        # Base random variation
        snr_noise = np.random.normal(0, 1.0)
        bw_factor = np.random.uniform(0.9, 1.1)

        # Jamming Event on HF Link (Steps 400 - 499)
        if self.name == "HF" and 400 <= step_num <= 499:
            self.current_snr_db = max(self.snr_min_db - 5.0, self.snr_min_db + snr_noise - 15.0)
            self.current_bw_bps = self.bw_min_bps
        # HF Partial Recovery (Steps 500 - 699)
        elif self.name == "HF" and 500 <= step_num <= 699:
            recovery_factor = (step_num - 500) / 200.0
            base_snr = np.random.uniform(self.snr_min_db, self.snr_max_db)
            self.current_snr_db = (self.snr_min_db - 5.0) + recovery_factor * (base_snr - (self.snr_min_db - 5.0))
            base_bw = np.random.uniform(self.bw_min_bps, self.bw_max_bps)
            self.current_bw_bps = self.bw_min_bps + recovery_factor * (base_bw - self.bw_min_bps)
        else:
            # Normal fluctuation
            raw_snr = self.current_snr_db + snr_noise
            self.current_snr_db = float(np.clip(raw_snr, self.snr_min_db, self.snr_max_db))
            raw_bw = self.current_bw_bps * bw_factor
            self.current_bw_bps = float(np.clip(raw_bw, self.bw_min_bps, self.bw_max_bps))

        self.current_ber = self._calculate_ber(self.current_snr_db)


def create_military_links() -> Dict[str, TacticalLink]:
    """Instantiate the 4 standard military tactical communication links."""
    return {
        "VLF": TacticalLink(
            name="VLF",
            mil_std="MIL-STD-188-141",
            bw_range_bps=(50.0, 300.0),
            snr_range_db=(5.0, 15.0),
            latency_budget_sec=30.0,
            clock_speed_mhz=24.0,  # ARM Cortex-M4
            use_case="Submarine / Deep-Bunker Comms"
        ),
        "HF": TacticalLink(
            name="HF",
            mil_std="MIL-STD-188-110D",
            bw_range_bps=(1200.0, 9600.0),
            snr_range_db=(10.0, 25.0),
            latency_budget_sec=5.0,
            clock_speed_mhz=72.0,  # ARM Cortex-M4
            use_case="Ground Tactical / NVIS"
        ),
        "UHF": TacticalLink(
            name="UHF",
            mil_std="MIL-STD-188-181",
            bw_range_bps=(16000.0, 256000.0),
            snr_range_db=(15.0, 35.0),
            latency_budget_sec=0.50,
            clock_speed_mhz=1200.0, # ARM Cortex-A53
            use_case="Vehicle SDR / Squad Radio"
        ),
        "SATCOM": TacticalLink(
            name="SATCOM",
            mil_std="MIL-STD-188-165",
            bw_range_bps=(512000.0, 2000000.0),
            snr_range_db=(8.0, 20.0),
            latency_budget_sec=0.80, # Includes ~270ms propagation each way
            clock_speed_mhz=1800.0, # ARM Cortex-A72
            use_case="Command Post Uplink / UAV Datalink"
        )
    }
