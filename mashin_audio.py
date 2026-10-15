"""Bounded, in-memory Mashin chunk decoding and default-device playback."""

from __future__ import annotations

import io
import logging
import struct
import threading
import time
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
DEFAULT_BUFFER_CAPACITY = 4 * 1024 * 1024
DECODE_AHEAD_CHUNKS = 2
LOGGER = logging.getLogger("mashin.audio")
TRACE = 5


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
class EncodedChunk:
    stream_id: str
    index: int
    first: int
    count: int
    final: bool
    data: bytes


@dataclass
class Stream:
    descriptor: dict[str, Any]
    chunks: deque[EncodedChunk] = field(default_factory=deque)
    decoded: deque[Chunk] = field(default_factory=deque)
    decoded_index: int = 0
    decode_failed: bool = False
    buffered_bytes: int = 0
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


def parse_chunk(
    data: bytes, stream: Stream, *, validate_order: bool = True
) -> EncodedChunk:
    """Validate chunk framing without decoding or changing the receipt cursor."""
    if not HEADER.size < len(data) <= HEADER.size + MAX_PAYLOAD:
        raise AudioError("Invalid audio message length")
    raw_id, index, first, count, final = HEADER.unpack_from(data)
    identifier = str(UUID(bytes=raw_id))
    if (
        identifier != stream.identifier
        or (validate_order and index != stream.next_index)
        or (validate_order and first != stream.received_frames)
        or not 1 <= count <= stream.rate
        or final not in (0, 1)
        or (not final and count != stream.rate)
        or (validate_order and stream.final)
    ):
        raise AudioError("Non-contiguous or invalid audio chunk")
    return EncodedChunk(identifier, index, first, count, bool(final), data)


def decode_chunk(data: bytes, stream: Stream, *, validate_order: bool = True) -> Chunk:
    """Decode one complete independent FLAC/Ogg mini-stream, including Opus pre-skip."""
    encoded = parse_chunk(data, stream, validate_order=validate_order)
    count = encoded.count
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
    return Chunk(
        encoded.stream_id, encoded.index, encoded.first, count, encoded.final, pcm
    )


class Output:
    """Retain encoded chunks and decode a bounded working set for PortAudio."""

    def __init__(
        self,
        device: str | int | None = None,
        buffer_capacity: int = DEFAULT_BUFFER_CAPACITY,
    ) -> None:
        if buffer_capacity < HEADER.size + MAX_PAYLOAD:
            raise AudioError(
                "Chunk buffer must hold at least one maximum-size audio message"
            )
        self.device = device
        self.buffer_capacity = buffer_capacity
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.decoder: threading.Thread | None = None
        self.decoder_stopping = False
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
        LOGGER.info(
            "Output opened: device=%s, rate=%s Hz, channels=%s",
            sd.query_devices(self.output.device, "output")["name"],
            stream.rate,
            stream.channels,
        )
        with self.condition:
            if self.decoder is None:
                self.decoder_stopping = False
                self.decoder = threading.Thread(
                    target=self._decode_worker, name="mashin-decoder", daemon=True
                )
                self.decoder.start()
            self.condition.notify_all()

    def load(self, descriptor: dict[str, Any], autoplay: bool) -> None:
        stream = stream_from(descriptor)
        self._stop_decoder()
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
        self._stop_decoder()
        self.close_device()
        with self.lock:
            self.current = self.next = None
            self.playing = False
            self.next_revision = 0
            self.scheduled.clear()

    def set_playing(self, playing: bool) -> None:
        with self.condition:
            self.playing = playing
            self.condition.notify_all()
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

    def buffered_bytes(self) -> int:
        with self.lock:
            return sum(
                stream.buffered_bytes
                for stream in (self.current, self.next)
                if stream is not None
            )

    def buffer_stats(self) -> tuple[int, int, float]:
        """Return cached wire bytes, chunk count and remaining output duration."""
        with self.lock:
            streams = [
                stream for stream in (self.current, self.next) if stream is not None
            ]
            chunks = sum(len(stream.chunks) for stream in streams)
            seconds = sum(
                sum(
                    max(0, chunk.count - max(0, stream.rendered - chunk.first))
                    for chunk in stream.chunks
                )
                / stream.rate
                for stream in streams
            )
            return self.buffered_bytes(), chunks, seconds

    def capacity(self) -> int:
        with self.lock:
            return max(
                0,
                min(
                    MAX_PAYLOAD,
                    self.buffer_capacity - self.buffered_bytes() - HEADER.size,
                ),
            )

    def accept(self, stream: Stream, data: bytes) -> EncodedChunk:
        with self.condition:
            if stream is not self.current and stream is not self.next:
                raise AudioError("Stream was replaced before retaining the chunk")
            chunk = parse_chunk(data, stream)
            if len(data) - HEADER.size > self.capacity():
                raise AudioError("Encoded chunk buffer is full")
            stream.chunks.append(chunk)
            stream.buffered_bytes += len(data)
            stream.next_index += 1
            stream.received_frames += chunk.count
            stream.final = chunk.final
            self.condition.notify_all()
            return chunk

    def prepare(self) -> bool:
        """Decode at most one buffered chunk without changing receipt authority."""
        with self.lock:
            candidate = self._decode_candidate()
            if candidate is None:
                return False
            stream, encoded = candidate
        started = time.monotonic()
        try:
            decoded = decode_chunk(encoded.data, stream, validate_order=False)
        except AudioError as failure:
            with self.condition:
                if self.current is stream:
                    stream.decode_failed = True
                    stream.error = {
                        "code": "stream_failed",
                        "message": "Buffered audio could not be decoded",
                        "is_retryable": False,
                        "retry_after_millis": None,
                    }
                    stream.chunks = deque(
                        chunk
                        for chunk in stream.chunks
                        if chunk.index <= stream.decoded_index
                    )
                    stream.buffered_bytes = sum(
                        len(chunk.data) for chunk in stream.chunks
                    )
                    self.condition.notify_all()
            LOGGER.error(
                "Buffered chunk decode failed: stream=%s index=%s: %s",
                stream.identifier,
                encoded.index,
                failure,
                exc_info=LOGGER.isEnabledFor(logging.DEBUG),
            )
            return False
        with self.condition:
            if self.current is stream and not stream.decode_failed:
                stream.decoded.append(decoded)
                stream.decoded_index = encoded.index
                self.condition.notify_all()
        LOGGER.log(
            TRACE,
            "Decoded for output: stream=%s index=%s frames=%s decode_ms=%.1f",
            stream.identifier,
            encoded.index,
            decoded.count,
            (time.monotonic() - started) * 1000,
        )
        return True

    def _decode_candidate(self) -> tuple[Stream, EncodedChunk] | None:
        stream = self.current
        if (
            stream is None
            or not self.playing
            or stream.decode_failed
            or len(stream.decoded) >= DECODE_AHEAD_CHUNKS
        ):
            return None
        encoded = next(
            (chunk for chunk in stream.chunks if chunk.index > stream.decoded_index),
            None,
        )
        return (stream, encoded) if encoded is not None else None

    def _decode_worker(self) -> None:
        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: (
                        self.decoder_stopping or self._decode_candidate() is not None
                    )
                )
                if self.decoder_stopping:
                    return
            self.prepare()

    def _stop_decoder(self) -> None:
        with self.condition:
            self.decoder_stopping = True
            self.condition.notify_all()
            decoder, self.decoder = self.decoder, None
        if decoder is not None:
            decoder.join()

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
                    and not stream.decoded
                    and not self.scheduled
                ):
                    state = "error"
                elif stream.ended:
                    state = "ended"
                elif not self.playing:
                    state = "paused"
                elif stream.decoded or self.scheduled:
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
                and not stream.decoded
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
        with self.condition:
            stream = self.current
            if stream is None or not self.playing or stream.ended:
                return
            cursor = 0
            while cursor < count and stream.decoded:
                chunk = stream.decoded[0]
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
                    stream.decoded.popleft()
                    stream.buffered_bytes -= len(stream.chunks.popleft().data)
                    self.condition.notify_all()
            if status and not status.output_underflow:
                self.failure = "Audio output reported a device error"
