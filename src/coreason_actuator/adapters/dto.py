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

from pydantic import BaseModel, ConfigDict, Field


class LocalEvictionPolicy(BaseModel):
    """
    AGENT INSTRUCTION: Internal DTO representing an EvictionPolicy for Actuator usage.
    """

    model_config = ConfigDict(extra="ignore")

    strategy: str
    max_retained_tokens: int
    protected_event_ids: list[str] = Field(default_factory=list)


class LocalToolManifest(BaseModel):
    """
    AGENT INSTRUCTION: Internal DTO representing a ToolManifest for Actuator usage.
    """

    model_config = ConfigDict(extra="ignore")

    tool_name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    is_preemptible: bool = False


class LocalToolInvocation(BaseModel):
    """
    AGENT INSTRUCTION: Internal DTO representing a ToolInvocationEvent combined with
    its execution context (manifest, eviction policy) for Actuator usage.
    """

    model_config = ConfigDict(extra="ignore")

    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    manifest: LocalToolManifest
    eviction_policy: LocalEvictionPolicy | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LocalToolInvocation":
        """Factory method to parse a generic dictionary payload into the DTO."""
        return cls.model_validate(payload)
