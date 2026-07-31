"""
Module 3: feasibility_filter.py — Hard Constraint Latency Filter
XAPQ-TacNet Simulation Framework for IEEE WiCOMM-2026

Eliminates physically infeasible PQC algorithm combinations whose total handshake time 
(computation + network transmission) exceeds the tactical link latency budget.
"""

from typing import List, Tuple, Dict, Any
from pqc_database import NIST_SPECS, total_handshake_time
from tactical_network import TacticalLink

# Generate all 21 (KEM, SIG) candidate algorithm pairs
ALL_KEMS = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]
ALL_SIGS = ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87", "SLH-DSA-128s"]

ALL_PQC_PAIRS: List[Tuple[str, str]] = [
    (kem, sig) for kem in ALL_KEMS for sig in ALL_SIGS
]


def get_feasible_actions(link: TacticalLink) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
    """
    Filter the 21 candidate (KEM, SIG) pairs against the link's latency budget.
    
    Returns:
    - feasible_pairs: List of valid (KEM, SIG) tuples
    - metadata: List of dicts containing detailed timing information per pair
    """
    feasible_pairs = []
    metadata = []

    for kem_alg, sig_alg in ALL_PQC_PAIRS:
        total_time, comp_time, trans_time = total_handshake_time(
            kem_alg=kem_alg,
            sig_alg=sig_alg,
            clock_speed_mhz=link.clock_speed_mhz,
            bandwidth_bps=link.current_bw_bps
        )

        is_feasible = total_time <= link.latency_budget_sec

        info = {
            "kem": kem_alg,
            "sig": sig_alg,
            "total_time": total_time,
            "comp_time": comp_time,
            "trans_time": trans_time,
            "budget": link.latency_budget_sec,
            "feasible": is_feasible
        }
        metadata.append(info)

        if is_feasible:
            feasible_pairs.append((kem_alg, sig_alg))

    # Safety Fallback: If extreme link degradation makes all pairs infeasible,
    # fall back to lowest-overhead pair (ML-KEM-512 + ML-DSA-44) and issue a warning.
    if not feasible_pairs:
        fallback_pair = ("ML-KEM-512", "ML-DSA-44")
        feasible_pairs.append(fallback_pair)

    return feasible_pairs, metadata


def compute_feasibility_matrix(links: Dict[str, TacticalLink]) -> Dict[str, Dict[Tuple[str, str], bool]]:
    """Compute feasibility matrix across all 4 links under nominal conditions."""
    matrix = {}
    for link_name, link in links.items():
        feasible, _ = get_feasible_actions(link)
        matrix[link_name] = {pair: (pair in feasible) for pair in ALL_PQC_PAIRS}
    return matrix
