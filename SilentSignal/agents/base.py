from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentMetadata:
    name: str
    description: str


class BaseAgent:
    metadata: AgentMetadata

    def __init__(self, metadata: AgentMetadata) -> None:
        self.metadata = metadata
