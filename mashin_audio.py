"""Bounded, in-memory Mashin chunk decoding and default-device playback."""

from __future__ import annotations

import io
import struct
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import av
import numpy as np
import sounddevice as sd

PROFILES = ("flac", "opus_96k", "opus_128k", "opus_192k", "opus_256k")
HEADER = struct.Struct("!16sQQQB")
MAX_PAYLOAD = 262144
MAX_PCM = 4 * 1024 * 1024
MAX_CHUNK_PCM = 48000 * 2 * 4


class AudioError(ValueError):
    """A stream, chunk or audio device cannot be played safely."""


@dataclass
class Chunk:
    stream_id: str
    index: int
    first: int
    count: int
    final: bool
    pcm: np.ndarray


@dataclass
class Stream:
    descriptor: dict[str, Any]
    chunks: deque[Chunk] = field(default_factory=deque)
    next_index: int = 1
    received_frames: int = 0
    rendered: int = 0
    position: int = 0
    final: bool = False
    ended: bool = False
    error: dict[str, Any] | None = None

    @property
    def identifier(self) -> str:
        return self.descriptor["stream_id"]

    @property
    def rate(self) -> int:
        return self.descriptor["output_format"]["sample_rate_hz"]

    @property
    def channels(self) -> int:
        return self.descriptor["output_format"]["channels"]


def stream_from(descriptor: dict[str, Any]) -> Stream:
    """Validate a descriptor before acknowledging its playback command."""
    try:
        identifier = UUID(descriptor["stream_id"])
        output = descriptor["output_format"]
        profile = output["profile_id"]
        valid = identifier.version == 4 and str(identifier) == descriptor["stream_id"]
        if profile == "flac":
            valid &= (
                output["container"] == output["codec"] == "flac"
                and output["sample_rate_hz"] in (44100, 48000)
                and output["channels"] in (1, 2)
                and output["bit_depth"] == 16
                and output["target_bitrate_bps"] is None
                and output["uses_vbr"] is False
            )
        else:
            valid &= (
                profile in PROFILES
                and output["container"] == "ogg"
                and output["codec"] == "opus"
                and output["sample_rate_hz"] == 48000
                and output["channels"] == 2
                and output["bit_depth"] is None
                and output["target_bitrate_bps"] == int(profile[5:-1]) * 1000
                and output["uses_vbr"] is True
            )
        if not valid:
            raise AudioError("Unsupported stream format")
    except (KeyError, TypeError, ValueError) as error:
        raise AudioError("Invalid stream descriptor") from error
    return Stream(descriptor)


def decode_chunk(data: bytes, stream: Stream) -> Chunk:
    """Decode one complete independent FLAC/Ogg mini-stream, including Opus pre-skip."""
    if not HEADER.size < len(data) <= HEADER.size + MAX_PAYLOAD:
        raise AudioError("Invalid audio message length")
    raw_id, index, first, count, final = HEADER.unpack_from(data)
    identifier = str(UUID(bytes=raw_id))
    if (
        identifier != stream.identifier
        or index != stream.next_index
        or first != stream.received_frames
        or not 1 <= count <= stream.rate
        or final not in (0, 1)
        or (not final and count != stream.rate)
        or stream.final
    ):
        raise AudioError("Non-contiguous or invalid audio chunk")
    frames: list[np.ndarray] = []
    decoded_count = 0
    output = stream.descriptor["output_format"]
    try:
        with av.open(
            io.BytesIO(data[HEADER.size :]), mode="r", format=output["container"]
        ) as container:
            resampler = av.AudioResampler(
                format="flt",
                layout="mono" if stream.channels == 1 else "stereo",
                rate=stream.rate,
            )
            for frame in container.decode(audio=0):
                if (
                    frame.sample_rate != stream.rate
                    or len(frame.layout.channels) != stream.channels
                ):
                    raise AudioError("Decoded format differs from descriptor")
                for converted in resampler.resample(frame):
                    decoded_count += converted.samples
                    if decoded_count > count:
                        raise AudioError("Decoded audio exceeds declared frame count")
                    frames.append(converted.to_ndarray().reshape(-1, stream.channels))
            for converted in resampler.resample(None):
                decoded_count += converted.samples
                frames.append(converted.to_ndarray().reshape(-1, stream.channels))
    except av.FFmpegError as error:
        raise AudioError("Invalid encoded audio") from error
    if decoded_count != count or not frames:
        raise AudioError("Decoded frame count differs from chunk header")
    pcm = np.concatenate(frames).astype(np.float32, copy=False)
    if not np.isfinite(pcm).all():
        raise AudioError("Non-finite audio samples")
    return Chunk(identifier, index, first, count, bool(final), pcm)


class Output:
    """Own one decoder queue and one PortAudio output stream."""

    def __init__(self, device: str | int | None = None) -> None:
        self.device = device
        self.lock = threading.RLock()
        self.current: Stream | None = None
        self.next: Stream | None = None
        self.next_revision = 0
        self.playing = False
        self.volume = 100
        self.muted = False
        self.output: sd.OutputStream | None = None
        self.scheduled: deque[tuple[float, int, int]] = deque()
        self.failure: str | None = None

    def _open(self, stream: Stream) -> None:
        self.close_device()
        sd.check_output_settings(
            device=self.device,
            channels=stream.channels,
            dtype="float32",
            samplerate=stream.rate,
        )
        self.output = sd.OutputStream(
            device=self.device,
            channels=stream.channels,
            dtype="float32",
            samplerate=stream.rate,
            callback=self._callback,
            latency="low",
        )
        self.output.start()

    def load(self, descriptor: dict[str, Any], autoplay: bool) -> None:
        stream = stream_from(descriptor)
        self.close_device()
        with self.lock:
            self.current, self.next = stream, None
            self.next_revision = 0
            self.playing = autoplay
            self.scheduled.clear()
            self.failure = None
        try:
            self._open(stream)
        except (sd.PortAudioError, ValueError):
            self.stop()
            raise

    def stop(self) -> None:
        self.close_device()
        with self.lock:
            self.current = self.next = None
            self.playing = False
            self.next_revision = 0
            self.scheduled.clear()

    def set_playing(self, playing: bool) -> None:
        with self.lock:
            self.playing = playing
        if self.output is not None:
            if playing:
                if not self.output.active:
                    self.output.start()
            else:
                self.output.stop()
                with self.lock:
                    if self.current is not None:
                        self.current.position = self.current.rendered
                    self.scheduled.clear()

    def close_device(self) -> None:
        output, self.output = self.output, None
        if output is not None:
            try:
                output.abort()
            finally:
                output.close()

    def capacity(self) -> int:
        with self.lock:
            used = sum(
                chunk.pcm.nbytes
                for stream in (self.current, self.next)
                if stream is not None
                for chunk in stream.chunks
            )
            return MAX_PAYLOAD if used + MAX_CHUNK_PCM <= MAX_PCM else 0

    def accept(self, stream: Stream, chunk: Chunk) -> None:
        with self.lock:
            if stream is not self.current and stream is not self.next:
                raise AudioError("Stream was replaced while decoding")
            if not self.capacity():
                raise AudioError("PCM buffer is full")
            stream.chunks.append(chunk)
            stream.next_index += 1
            stream.received_frames += chunk.count
            stream.final = chunk.final

    def cursors(self) -> list[dict[str, Any]]:
        with self.lock:
            return [
                {"stream_id": stream.identifier, "next_chunk_index": stream.next_index}
                for stream in (self.current, self.next)
                if stream is not None
            ]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            stream = self.current
            state = "idle"
            if stream is not None:
                if (
                    stream.error is not None
                    and not stream.chunks
                    and not self.scheduled
                ):
                    state = "error"
                elif stream.ended:
                    state = "ended"
                elif not self.playing:
                    state = "paused"
                elif stream.chunks or self.scheduled:
                    state = "playing"
                else:
                    state = "buffering"
            return {
                "stream_id": stream.identifier if stream else None,
                "output_revision": 0,
                "output_state": state,
                "position_frame": stream.position if stream else 0,
                "volume_percent": self.volume,
                "is_muted": self.muted,
                "error": stream.error if stream else None,
            }

    def poll(self) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        with self.lock:
            stream = self.current
            if stream is None or self.output is None:
                return events
            now = self.output.time
            while self.scheduled:
                started, first, count = self.scheduled[0]
                heard = min(count, max(0, int((now - started) * stream.rate)))
                stream.position = max(stream.position, first + heard)
                if heard < count:
                    break
                self.scheduled.popleft()
            if (
                self.playing
                and stream.final
                and not stream.chunks
                and not self.scheduled
                and not stream.ended
                and stream.error is None
            ):
                stream.ended = True
                events.append(("track_ended", {"stream_id": stream.identifier}))
            promote = (
                self.playing
                and stream.ended
                and self.next is not None
                and bool(self.next.chunks)
            )
        if promote:
            self.close_device()
            with self.lock:
                self.current, self.next = self.next, None
                self.scheduled.clear()
                stream = self.current
                assert stream is not None
                events.append(
                    (
                        "track_started",
                        {
                            "stream_id": stream.identifier,
                            "next_revision": self.next_revision,
                            "output_revision": 0,
                            "position_frame": 0,
                        },
                    )
                )
            self._open(stream)
        return events

    def _callback(
        self, output: np.ndarray, count: int, clock: Any, status: Any
    ) -> None:
        output.fill(0)
        with self.lock:
            stream = self.current
            if stream is None or not self.playing or stream.ended:
                return
            cursor = 0
            while cursor < count and stream.chunks:
                chunk = stream.chunks[0]
                offset = stream.rendered - chunk.first
                take = min(count - cursor, chunk.count - offset)
                gain = 0.0 if self.muted else self.volume / 100.0
                output[cursor : cursor + take] = (
                    chunk.pcm[offset : offset + take] * gain
                )
                self.scheduled.append(
                    (
                        clock.outputBufferDacTime + cursor / stream.rate,
                        stream.rendered,
                        take,
                    )
                )
                stream.rendered += take
                cursor += take
                if offset + take == chunk.count:
                    stream.chunks.popleft()
            if status and not status.output_underflow:
                self.failure = "Audio output reported a device error"
