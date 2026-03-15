# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_actuator

import re


class MaskingFunctor:
    """
    A deterministic Regex/Aho-Corasick string replacement pass.
    Every unsealed string retrieved from the Vault must be mathematically
    replaced with [REDACTED_BY_ACTUATOR_SECURITY_BOUNDARY].
    """

    REDACTION_STRING = "[REDACTED_BY_ACTUATOR_SECURITY_BOUNDARY]"

    def __init__(self, secrets: list[str]) -> None:
        """
        Initializes the masking functor with a list of secrets to redact.
        Empty strings or None are filtered out mathematically to prevent matching everything.
        """
        self._secrets = [s for s in secrets if s and isinstance(s, str)]

        # Sort by length descending to ensure longer secrets are replaced before
        # shorter ones that might be a substring (e.g. "secret123" vs "secret")
        self._secrets.sort(key=len, reverse=True)

        self._pattern: re.Pattern[str] | None = None
        if self._secrets:
            # Escape secrets for regex and combine with OR
            pattern_str = "|".join(map(re.escape, self._secrets))
            self._pattern = re.compile(pattern_str)

    def redact(self, text: str) -> str:
        """
        Replaces all occurrences of the loaded secrets in the text with the redaction string.
        """
        if not text or not self._pattern:
            return text
        return self._pattern.sub(self.REDACTION_STRING, text)
