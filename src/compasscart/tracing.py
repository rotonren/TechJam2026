from __future__ import annotations

from collections import deque
from collections.abc import Mapping


class TraceSink:
    def __init__(self, max_entries: int = 5_000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._records: deque[dict[str, object]] = deque(maxlen=max_entries)
        self.enabled = True

    @property
    def records(self) -> list[dict[str, object]]:
        return list(self._records)

    def record(self, payload: Mapping[str, object]) -> None:
        if not self.enabled:
            return
        try:
            self._records.append(dict(payload))
        except Exception:  # noqa: BLE001 - tracing cannot affect scored responses.
            self.enabled = False
