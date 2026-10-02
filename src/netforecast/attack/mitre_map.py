"""MITRE ATT&CK Mapping Module.

Maps network attack labels (from CIC-IDS-2018 or other telemetry) to standardized
MITRE ATT&CK enterprise stages, tactics, and infiltration indicators.
"""

from typing import Any, Dict, List, Optional, Tuple
import yaml


DEFAULT_STAGE_NAMES = {
    0: "Benign",
    1: "Reconnaissance",
    2: "Initial Access",
    3: "Lateral Movement",
    4: "Command and Control",
    5: "Exfiltration/Impact",
}

DEFAULT_MAPPING = {
    "Benign": {
        "stage": "Benign",
        "stage_id": 0,
        "mitre_tactic": "None",
        "tactic_id": "TA0000",
        "is_infiltration": 0,
    },
    "FTP-BruteForce": {
        "stage": "Initial Access",
        "stage_id": 2,
        "mitre_tactic": "Credential Access / Initial Access",
        "tactic_id": "T1110",
        "is_infiltration": 1,
    },
    "SSH-Bruteforce": {
        "stage": "Initial Access",
        "stage_id": 2,
        "mitre_tactic": "Credential Access / Initial Access",
        "tactic_id": "T1110",
        "is_infiltration": 1,
    },
    "DoS-GoldenEye": {
        "stage": "Impact",
        "stage_id": 5,
        "mitre_tactic": "Denial of Service",
        "tactic_id": "T1498",
        "is_infiltration": 0,
    },
    "DoS-Slowloris": {
        "stage": "Impact",
        "stage_id": 5,
        "mitre_tactic": "Denial of Service",
        "tactic_id": "T1498",
        "is_infiltration": 0,
    },
    "Infiltration": {
        "stage": "Lateral Movement",
        "stage_id": 3,
        "mitre_tactic": "Lateral Movement / Exfiltration",
        "tactic_id": "T1021",
        "is_infiltration": 1,
    },
    "Bot": {
        "stage": "Command and Control",
        "stage_id": 4,
        "mitre_tactic": "Command and Control",
        "tactic_id": "T1071",
        "is_infiltration": 1,
    },
}


class MitreMapper:
    """Translates flow/packet labels into MITRE ATT&CK stages and tactics."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        if config_dict and "mitre_attack_mapping" in config_dict:
            self.mapping = config_dict["mitre_attack_mapping"]
        else:
            self.mapping = DEFAULT_MAPPING

        if config_dict and "stages" in config_dict:
            self.stage_names = {
                int(k): v for k, v in config_dict["stages"].items()
            }
        else:
            self.stage_names = DEFAULT_STAGE_NAMES

    def get_stage_info(self, raw_label: str) -> Dict[str, Any]:
        """Look up MITRE ATT&CK details for a raw dataset label."""
        cleaned = str(raw_label).strip()

        # Direct match
        if cleaned in self.mapping:
            return self.mapping[cleaned]

        # Case-insensitive or partial match
        cleaned_lower = cleaned.lower()
        for k, v in self.mapping.items():
            if k.lower() == cleaned_lower:
                return v

        for k, v in self.mapping.items():
            if k.lower() in cleaned_lower:
                return v

        # Default to Benign if unknown
        return self.mapping.get(
            "Benign",
            {
                "stage": "Benign",
                "stage_id": 0,
                "mitre_tactic": "None",
                "tactic_id": "TA0000",
                "is_infiltration": 0,
            },
        )

    def get_stage_name(self, stage_id: int) -> str:
        """Return stage text from stage ID (0..5)."""
        return self.stage_names.get(stage_id, f"Unknown Stage ({stage_id})")

    @property
    def num_stages(self) -> int:
        """Total number of MITRE stages."""
        return len(self.stage_names)
