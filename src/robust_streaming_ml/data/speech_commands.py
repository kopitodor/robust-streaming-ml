"""Speech Commands v0.02, 10-keyword subset: contract parameters.

Values mirror docs/data-contract.md and docs/datasets/speech_commands_v0.02.md
(a test keeps them in sync with the verification record).
"""

from __future__ import annotations

from robust_streaming_ml.data.acquisition import SPEECH_COMMANDS_V0_02
from robust_streaming_ml.data.validation import DatasetSpec

__all__ = ["LABELS", "SPEECH_COMMANDS_KW10"]

LABELS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")

SPEECH_COMMANDS_KW10 = DatasetSpec(
    archive=SPEECH_COMMANDS_V0_02,
    labels=LABELS,
    manifest_name="speech_commands_v0.02_kw10",
    # First generated and verified 2026-09-27; see docs/datasets/speech_commands_v0.02.md.
    expected_manifest_sha256="f811a20d030ee415fced0b3bc8d7471d6ebdbb4b3d4ee680ec62d8614ab6131c",
    metadata={
        "retrieved_at_utc": "2026-09-27T18:45:33Z",
        "license_name": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "license_file_path": "LICENSE",
        "attribution_text": (
            "Speech Commands Data Set v0.02, by Pete Warden (Google), licensed under CC BY 4.0 "
            "(https://creativecommons.org/licenses/by/4.0/). Source: "
            "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz. Changes: only the "
            "10-keyword subset is used, and it is described by a derived manifest; no audio is "
            "redistributed."
        ),
        "citation": (
            "Warden, P. (2018). Speech Commands: A Dataset for Limited-Vocabulary Speech "
            "Recognition. arXiv:1804.03209"
        ),
        "redistribution_notes": (
            "CC BY 4.0 allows redistribution with attribution and an indication of changes. "
            "No audio is redistributed; this manifest carries the attribution above."
        ),
        "license_verified_by": "Project owner",
        "license_verified_on": "2026-09-27",
    },
)
