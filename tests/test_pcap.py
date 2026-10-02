"""Unit tests for Scapy packet-level feature extraction."""

import tempfile
from pathlib import Path
import pytest

from netforecast.data.pcap_features import create_sample_pcap, extract_features_from_pcap


def test_pcap_generation_and_parsing():
    """Verify PCAP generation and subsequent packet-level feature extraction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pcap_file = Path(tmpdir) / "test_capture.pcap"
        created_path = create_sample_pcap(pcap_file, num_packets=20)
        assert created_path.exists()

        df_pcap = extract_features_from_pcap(created_path, window_seconds=2.0)

        assert not df_pcap.empty
        assert "window_idx" in df_pcap.columns
        assert "ttl_mean" in df_pcap.columns
        assert "tcp_win_mean" in df_pcap.columns
        assert "port_scan_signature" in df_pcap.columns
        assert "syn_ratio" in df_pcap.columns
        assert len(df_pcap) > 0
