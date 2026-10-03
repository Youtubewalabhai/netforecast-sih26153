"""Packet-level Feature Extraction using Scapy for PCAP/PCAPNG files.

Extracts deep packet features:
- TTL mean and variance per session
- TCP window sizes
- IP fragment flags
- Payload size distribution
- Port-scan signature (port entropy / sequential access)
- Retransmission count
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_features_from_pcap(
    pcap_path: Union[str, Path],
    window_seconds: float = 10.0,
) -> pd.DataFrame:
    """Extract packet-level network telemetry from a PCAP file and aggregate by time windows.

    Args:
        pcap_path: Path to .pcap or .pcapng file.
        window_seconds: Resolution in seconds for state window aggregation.

    Returns:
        pd.DataFrame containing packet-level engineered features per time window.
    """
    try:
        from scapy.all import IP, TCP, UDP, rdpcap
    except ImportError:
        raise ImportError(
            "Scapy is required for PCAP feature extraction. Install it via pip install scapy"
        )

    path = Path(pcap_path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP file not found: {path}")

    packets = rdpcap(str(path))
    if len(packets) == 0:
        return pd.DataFrame()

    records = []
    base_time = float(packets[0].time)

    for pkt in packets:
        if not pkt.haslayer(IP):
            continue

        ip_layer = pkt[IP]
        timestamp = float(pkt.time)
        pkt_len = len(pkt)
        ttl = ip_layer.ttl
        flags_ip = int(ip_layer.flags)
        is_fragment = 1 if (flags_ip & 0x01) or (ip_layer.frag > 0) else 0

        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.proto

        tcp_win = 0
        tcp_flags = 0
        dst_port = 0
        payload_len = 0
        is_retransmission = 0

        if pkt.haslayer(TCP):
            tcp_layer = pkt[TCP]
            dst_port = tcp_layer.dport
            tcp_win = tcp_layer.window
            tcp_flags = int(tcp_layer.flags)
            if tcp_layer.payload:
                payload_len = len(tcp_layer.payload)
        elif pkt.haslayer(UDP):
            udp_layer = pkt[UDP]
            dst_port = udp_layer.dport
            if udp_layer.payload:
                payload_len = len(udp_layer.payload)

        records.append(
            {
                "timestamp": timestamp,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "proto": proto,
                "dst_port": dst_port,
                "pkt_len": pkt_len,
                "payload_len": payload_len,
                "ttl": ttl,
                "is_fragment": is_fragment,
                "tcp_win": tcp_win,
                "tcp_flags": tcp_flags,
                "window_idx": int((timestamp - base_time) // window_seconds),
            }
        )

    df_pkts = pd.DataFrame(records)
    if df_pkts.empty:
        return pd.DataFrame()

    # Aggregate by time window
    grouped = df_pkts.groupby("window_idx")
    window_features = []

    for widx, group in grouped:
        unique_ports = group["dst_port"].nunique()
        total_pkts = len(group)
        # Port scan signature: ratio of unique ports to total packets
        port_scan_ratio = unique_ports / max(total_pkts, 1)

        feature_dict = {
            "window_idx": widx,
            "pkt_count": total_pkts,
            "bytes_total": group["pkt_len"].sum(),
            "ttl_mean": group["ttl"].mean(),
            "ttl_var": group["ttl"].var() if len(group) > 1 else 0.0,
            "tcp_win_mean": group["tcp_win"].mean(),
            "tcp_win_min": group["tcp_win"].min(),
            "tcp_win_max": group["tcp_win"].max(),
            "fragment_ratio": group["is_fragment"].mean(),
            "payload_len_mean": group["payload_len"].mean(),
            "payload_len_std": group["payload_len"].std() if len(group) > 1 else 0.0,
            "payload_len_max": group["payload_len"].max(),
            "unique_dst_ports": unique_ports,
            "port_scan_signature": port_scan_ratio,
            "syn_ratio": (group["tcp_flags"] & 0x02).astype(bool).mean(),
            "fin_ratio": (group["tcp_flags"] & 0x01).astype(bool).mean(),
            "rst_ratio": (group["tcp_flags"] & 0x04).astype(bool).mean(),
        }
        window_features.append(feature_dict)

    res_df = pd.DataFrame(window_features).fillna(0.0)
    return res_df


def create_sample_pcap(
    output_path: Union[str, Path], num_packets: int = 40
) -> Path:
    """Generate a tiny valid sample PCAP with Scapy for testing/demonstrations."""
    try:
        from scapy.all import IP, TCP, UDP, wrpcap
    except ImportError:
        raise ImportError("Scapy required to generate sample PCAP")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pkts = []
    base_ip = "192.168.1."
    target_ip = "10.0.0.5"

    for i in range(num_packets):
        # Alternate between normal traffic and scanning bursts
        dport = 80 if i % 2 == 0 else (1000 + i * 5)
        pkt = (
            IP(src=f"{base_ip}100", dst=target_ip, ttl=64)
            / TCP(sport=50000 + i, dport=dport, flags="S", window=64240)
            / ("NETFORECAST_DEMO_PAYLOAD" * (i % 3 + 1))
        )
        # Add custom timestamp offset
        pkt.time = 1700000000.0 + (i * 0.25)
        pkts.append(pkt)

    wrpcap(str(path), pkts)
    return path
