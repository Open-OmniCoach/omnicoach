"""Shared deterministic test helpers for `omnicoach_harness` tests."""

from __future__ import annotations

from omnicoach_harness.providers.protocol import ModelRequest, ModelResponse


class QueuedModelClient:
    """Deterministic test client with full request capture."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.outputs:
            raise AssertionError("queued model client ran out of outputs")
        return ModelResponse(
            text=self.outputs.pop(0),
            model=request.model,
            provider="queued",
            metadata={"response_format": request.response_format},
        )
