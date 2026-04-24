"""Small screen-logging helpers with user-timezone timestamps."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TextIO


def user_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _coerce_message(message: str, args: tuple[object, ...]) -> str:
    if not args:
        return str(message)
    return str(message) % args


def format_screen_message(
    component: str,
    message: str,
    *,
    level: str = "INFO",
    elapsed: float | None = None,
    rank: int | None = None,
) -> str:
    header = [user_timestamp(), component, level.upper()]
    if rank is not None:
        header.append(f"rank={rank}")
    if elapsed is not None:
        header.append(f"+{elapsed:.1f}s")
    return f"[{' '.join(header)}] {message}"


@dataclass
class ScreenLogger:
    component: str
    start_time: float | None = None
    stream: TextIO = sys.stdout
    error_stream: TextIO = sys.stderr

    def _emit(
        self,
        level: str,
        message: str,
        *args: object,
        elapsed: float | None = None,
        rank: int | None = None,
        stream: TextIO | None = None,
    ) -> None:
        text = _coerce_message(message, args)
        if elapsed is None and self.start_time is not None:
            elapsed = time.monotonic() - self.start_time
        target = stream
        if target is None:
            target = self.error_stream if level.upper() == "ERROR" else self.stream
        print(
            format_screen_message(
                self.component,
                text,
                level=level,
                elapsed=elapsed,
                rank=rank,
            ),
            file=target,
            flush=True,
        )

    def info(
        self,
        message: str,
        *args: object,
        elapsed: float | None = None,
        rank: int | None = None,
    ) -> None:
        self._emit("INFO", message, *args, elapsed=elapsed, rank=rank)

    def warning(
        self,
        message: str,
        *args: object,
        elapsed: float | None = None,
        rank: int | None = None,
    ) -> None:
        self._emit("WARNING", message, *args, elapsed=elapsed, rank=rank)

    def error(
        self,
        message: str,
        *args: object,
        elapsed: float | None = None,
        rank: int | None = None,
    ) -> None:
        self._emit("ERROR", message, *args, elapsed=elapsed, rank=rank)


def get_screen_logger(
    component: str,
    *,
    start_time: float | None = None,
    stream: TextIO = sys.stdout,
    error_stream: TextIO = sys.stderr,
) -> ScreenLogger:
    return ScreenLogger(
        component=component,
        start_time=start_time,
        stream=stream,
        error_stream=error_stream,
    )
