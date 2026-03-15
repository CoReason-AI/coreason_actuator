# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_actuator

from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from coreason_actuator.security import MaskingFunctor


def test_masking_functor_basic() -> None:
    """Test standard replacement."""
    secrets = ["my_secret", "another_password"]
    functor = MaskingFunctor(secrets)

    original = "Hello my_secret world!"
    redacted = functor.redact(original)

    assert redacted == f"Hello {MaskingFunctor.REDACTION_STRING} world!"


def test_masking_functor_multiple_secrets() -> None:
    """Test replacing multiple secrets in the same text."""
    secrets = ["secretA", "secretB"]
    functor = MaskingFunctor(secrets)

    original = "Here is secretA and here is secretB"
    redacted = functor.redact(original)

    assert redacted == f"Here is {MaskingFunctor.REDACTION_STRING} and here is {MaskingFunctor.REDACTION_STRING}"


def test_masking_functor_substring_secrets() -> None:
    """Test that longer secrets matching substring of shorter secrets are replaced correctly."""
    secrets = ["secret", "secret123"]
    functor = MaskingFunctor(secrets)

    original = "The code is secret123 and also secret"
    redacted = functor.redact(original)

    assert redacted == f"The code is {MaskingFunctor.REDACTION_STRING} and also {MaskingFunctor.REDACTION_STRING}"


def test_masking_functor_no_secrets() -> None:
    """Test with empty secrets list."""
    functor = MaskingFunctor([])
    original = "Just some text."
    redacted = functor.redact(original)
    assert redacted == original


def test_masking_functor_none_text() -> None:
    """Test with None text."""
    functor = MaskingFunctor(["secret"])
    assert functor.redact(None) is None  # type: ignore[arg-type]
    assert functor.redact("") == ""


def test_masking_functor_invalid_secrets() -> None:
    """Test with invalid secrets (empty strings, None, integers)."""
    secrets: list[Any] = ["", None, "valid_secret", 123]
    functor = MaskingFunctor(secrets)

    original = "Text with valid_secret"
    redacted = functor.redact(original)

    assert redacted == f"Text with {MaskingFunctor.REDACTION_STRING}"


@given(
    text=st.text(),
    secrets=st.lists(st.text(min_size=1)),
)
def test_masking_functor_property(text: str, secrets: list[str]) -> None:
    """Property-based testing for the MaskingFunctor."""
    functor = MaskingFunctor(secrets)
    redacted = functor.redact(text)

    # We mathematically assert that no valid secret string remains in the redacted text
    # EXCEPT if the secret itself is a substring of the REDACTION_STRING.
    for secret in functor._secrets:
        if secret in text and secret not in MaskingFunctor.REDACTION_STRING:
            assert secret not in redacted


@given(
    prefix=st.text(),
    secret=st.text(min_size=1),
    suffix=st.text(),
)
def test_masking_functor_guaranteed_replacement(prefix: str, secret: str, suffix: str) -> None:
    """Property-based testing to guarantee replacement."""
    functor = MaskingFunctor([secret])
    original = prefix + secret + suffix
    redacted = functor.redact(original)

    # If the secret is part of the redaction string, it will naturally appear in the redacted text
    if secret not in MaskingFunctor.REDACTION_STRING:
        assert secret not in redacted

    # Redaction string should always be inserted if secret was present in original
    if secret in original:
        assert MaskingFunctor.REDACTION_STRING in redacted
