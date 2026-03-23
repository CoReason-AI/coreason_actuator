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

from coreason_actuator.adapters.dto import LocalEvictionPolicy, LocalToolInvocation, LocalToolManifest


def test_local_eviction_policy_ignores_extra() -> None:
    """Test that LocalEvictionPolicy silently drops unknown fields."""
    payload: dict[str, Any] = {
        "strategy": "fifo",
        "max_retained_tokens": 1000,
        "protected_event_ids": ["1", "2"],
        "unknown_field": "some_value",
        "another_extra_field": 42,
    }

    policy = LocalEvictionPolicy.model_validate(payload)

    assert policy.strategy == "fifo"
    assert policy.max_retained_tokens == 1000
    assert policy.protected_event_ids == ["1", "2"]
    assert not hasattr(policy, "unknown_field")
    assert not hasattr(policy, "another_extra_field")


def test_local_tool_manifest_ignores_extra() -> None:
    """Test that LocalToolManifest silently drops unknown fields."""
    payload: dict[str, Any] = {
        "tool_name": "example_tool",
        "description": "An example tool",
        "input_schema": {"type": "object"},
        "is_preemptible": True,
        "future_field_we_dont_know_about": ["123"],
    }

    manifest = LocalToolManifest.model_validate(payload)

    assert manifest.tool_name == "example_tool"
    assert manifest.description == "An example tool"
    assert manifest.input_schema == {"type": "object"}
    assert manifest.is_preemptible is True
    assert not hasattr(manifest, "future_field_we_dont_know_about")


def test_local_tool_invocation_from_payload() -> None:
    """Test that LocalToolInvocation correctly parses a nested payload and ignores extra fields."""
    payload: dict[str, Any] = {
        "tool_name": "example_tool",
        "parameters": {"arg1": "value1"},
        "manifest": {
            "tool_name": "example_tool",
            "description": "An example tool",
            "input_schema": {},
            "is_preemptible": False,
            "extra_manifest_field": "should be ignored",
        },
        "eviction_policy": {
            "strategy": "salience_decay",
            "max_retained_tokens": 500,
            "extra_policy_field": "should be ignored",
        },
        "extra_invocation_field": "should be ignored",
    }

    invocation = LocalToolInvocation.from_payload(payload)

    assert invocation.tool_name == "example_tool"
    assert invocation.parameters == {"arg1": "value1"}

    assert isinstance(invocation.manifest, LocalToolManifest)
    assert invocation.manifest.tool_name == "example_tool"
    assert invocation.manifest.description == "An example tool"
    assert invocation.manifest.input_schema == {}
    assert invocation.manifest.is_preemptible is False
    assert not hasattr(invocation.manifest, "extra_manifest_field")

    assert isinstance(invocation.eviction_policy, LocalEvictionPolicy)
    assert invocation.eviction_policy.strategy == "salience_decay"
    assert invocation.eviction_policy.max_retained_tokens == 500
    assert invocation.eviction_policy.protected_event_ids == []
    assert not hasattr(invocation.eviction_policy, "extra_policy_field")

    assert not hasattr(invocation, "extra_invocation_field")


def test_local_tool_invocation_without_eviction_policy() -> None:
    """Test that LocalToolInvocation correctly parses when eviction_policy is missing."""
    payload: dict[str, Any] = {
        "tool_name": "minimal_tool",
        "parameters": {},
        "manifest": {
            "tool_name": "minimal_tool",
        },
    }

    invocation = LocalToolInvocation.from_payload(payload)

    assert invocation.tool_name == "minimal_tool"
    assert invocation.parameters == {}

    assert isinstance(invocation.manifest, LocalToolManifest)
    assert invocation.manifest.tool_name == "minimal_tool"
    assert invocation.manifest.description == ""
    assert invocation.manifest.input_schema == {}
    assert invocation.manifest.is_preemptible is False

    assert invocation.eviction_policy is None
