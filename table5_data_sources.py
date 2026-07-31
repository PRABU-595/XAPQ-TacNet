"""
table5_data_sources.py — Table 5: Real-World Data Sources Used in Simulation
XAPQ-TacNet Simulation Framework for IEEE WiCOMM-2026
"""

def print_table5():
    sources = [
        ("KEM byte sizes", "NIST FIPS 203 (ML-KEM Standard)", "Mathematical constant"),
        ("Signature byte sizes", "NIST FIPS 204/205 (ML-DSA/SLH-DSA Standards)", "Mathematical constant"),
        ("PQC cycles (Cortex-M4)", "pqm4 (Kannwischer et al., ePrint 2019/844)", "Hardware measurement"),
        ("PQC TLS handshake", "Fitzgibbon & Ottaviani, Cryptography 8(2), 2024", "Hardware measurement"),
        ("PQC on RPi 3B+/5", "Lopez et al., arXiv:2507.08312, IEEE QCE 2025", "Hardware measurement"),
        ("VLF link specs", "MIL-STD-188-141", "Military standard"),
        ("HF link specs", "MIL-STD-188-110D", "Military standard"),
        ("UHF link specs", "MIL-STD-188-181", "Military standard"),
        ("SATCOM link specs", "MIL-STD-188-165", "Military standard"),
        ("BER model", "Proakis, Digital Communications, 5th ed., Ch. 5", "Textbook physics"),
        ("LinUCB algorithm", "Li et al., WWW 2010 (4000+ citations)", "Peer-reviewed method"),
        ("DoW PQC deadlines", "DoW PQC Strategy, June 2026 (EO 14347)", "Policy document")
    ]

    print("\n" + "=" * 85)
    print("TABLE 5: Real-World Verifiable Data Sources Used in XAPQ-TacNet Simulation")
    print("=" * 85)
    header = f"{'Component':<24} | {'Source Citation':<45} | {'Data Type':<20}"
    print(header)
    print("-" * len(header))
    for comp, src, dtype in sources:
        print(f"{comp:<24} | {src:<45} | {dtype:<20}")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    print_table5()
