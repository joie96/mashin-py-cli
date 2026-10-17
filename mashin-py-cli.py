# /// script
# requires-python = ">=3.12"
# dependencies = ["aiohttp>=3.12,<4", "av>=19,<20", "numpy>=2,<3", "sounddevice>=0.5.6,<0.6"]
# ///
"""Mashin V3 console receiver. Playback is controlled exclusively through MA."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import secrets
import socket
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

import aiohttp
import sounddevice as sd
from mashin_audio import (
    HEADER,
    MAX_PAYLOAD,
    PROFILES,
    TRACE,
    AudioError,
    Output,
)

LOGGER = logging.getLogger("mashin")
PROTOCOL_VERSION = 3
logging.addLevelName(TRACE, "TRACE")
COMMANDS = {
    "set_media",
    "play",
    "pause",
    "stop",
    "set_volume",
    "set_mute",
    "state_request",
}


def server_url(value: str) -> str:
    """Normalize an MA base URL without accepting credentials or query parameters."""
    parsed = urlsplit(value if "://" in value else "http://" + value)
    if (
        parsed.scheme not in ("http", "https", "ws", "wss")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise argparse.ArgumentTypeError(
            "Use an HTTP(S)/WS(S) server URL without credentials or query"
        )
    try:
        _ = parsed.port
    except ValueError as error:
        raise argparse.ArgumentTypeError("Invalid server port") from error
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    return urlunsplit((scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def protocol_error(code: str, message: str) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "is_retryable": False,
        "retry_after_millis": None,
    }


class Client:
    """Keep Control and Audio transport ownership separate from device playback."""

    def __init__(
        self, arguments: argparse.Namespace, output: Output | None = None
    ) -> None:
        self.arguments = arguments
        self.output = output or Output(
            arguments.output_device, arguments.buffer_capacity * 1024 * 1024
        )
        self.control: aiohttp.ClientWebSocketResponse | None = None
        self.audio: aiohttp.ClientWebSocketResponse | None = None
        self.epoch: str | None = None
        self.sequence = 0
        self.audio_id = 1
        self.send_lock = asyncio.Lock()
        self.ready = asyncio.Event()
        self.pong = asyncio.Event()
        self.ping_payload = b""
        self.acks: OrderedDict[int, tuple[str, dict[str, Any], dict[str, Any]]] = (
            OrderedDict()
        )
        self.last_capacity: int | None = None
        self.output_lock = asyncio.Lock()
        self.disconnected_at: float | None = None
        self.last_buffer_summary = time.monotonic()
        self.last_state: tuple[Any, ...] | None = None

    async def operate(self, operation: Callable[..., Any], *args: Any) -> Any:
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def send(
        self, kind: str, payload: dict[str, Any], *, sequence: bool = False
    ) -> None:
        async with self.send_lock:
            if self.control is None or self.control.closed:
                raise ConnectionError("Control disconnected")
            if sequence:
                self.sequence += 1
                payload = {"sequence": self.sequence, **payload}
            await self.control.send_json(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "type": kind,
                    "message_id": str(uuid4()),
                    "connection_epoch": self.epoch,
                    "payload": payload,
                },
                dumps=lambda value: json.dumps(
                    value, allow_nan=False, separators=(",", ":")
                ),
            )

    def resume(self) -> dict[str, Any]:
        state = self.output.snapshot()
        state.pop("error")
        state["current_stream_id"] = state.pop("stream_id")
        return {
            **state,
            "media_revision": self.output.media_revision,
            "received_streams": self.output.cursors(),
            "activation": asdict(self.output.activation)
            if self.output.activation
            else None,
        }

    async def run(self) -> None:
        LOGGER.info(
            "Receiver starting: device_id=%s, name=%s, profile=%s, encoded chunk buffer=%.1f MiB, output=%s",
            self.arguments.device_id,
            self.arguments.name,
            self.arguments.profile,
            self.output.buffer_capacity / (1024 * 1024),
            self.arguments.output_device
            if self.arguments.output_device is not None
            else "default",
        )
        delay = 0.5
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
        monitor = asyncio.create_task(self.buffer_loop())
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                while True:
                    try:
                        await self.connect(session)
                    except (
                        aiohttp.ClientConnectorCertificateError,
                        aiohttp.WSServerHandshakeError,
                    ) as error:
                        raise AudioError(
                            "WebSocket upgrade or TLS verification failed"
                        ) from error
                    except (
                        aiohttp.ClientError,
                        ConnectionError,
                        TimeoutError,
                        OSError,
                    ) as error:
                        if self.disconnected_at is None:
                            self.disconnected_at = time.monotonic()
                            delay = 0.5
                        LOGGER.warning(
                            "Connection lost (%s: %s); reconnecting in %.1f s",
                            type(error).__name__,
                            error,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        delay = min(8, delay * 2)
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            await self.operate(self.output.stop)
            LOGGER.info("Receiver stopped; audio device closed")

    async def buffer_loop(self) -> None:
        while True:
            await asyncio.sleep(0.25)
            if self.disconnected_at is not None:
                async with self.output_lock:
                    events = await self.operate(self.output.poll)
                for kind, payload in events:
                    LOGGER.info(
                        "Offline playback event: %s stream=%s",
                        kind,
                        payload["stream_id"],
                    )
            self.log_buffer_summary(time.monotonic())

    async def connect(self, session: aiohttp.ClientSession) -> None:
        await self.operate(self.output.poll)
        self.epoch, self.sequence = None, 0
        self.acks.clear()
        self.ready.clear()
        self.last_capacity = None
        self.last_state = None
        tasks: list[asyncio.Task[None]] = []
        ws_timeout = aiohttp.ClientWSTimeout(ws_close=2)
        try:
            LOGGER.info(
                "Connecting Control to %s", urlsplit(self.arguments.server).netloc
            )
            self.control = await session.ws_connect(
                self.arguments.server + "/mashin/v1/control",
                autoping=False,
                max_msg_size=65536,
                timeout=ws_timeout,
            )
            await self.send(
                "hello",
                {
                    "device_id": self.arguments.device_id,
                    "client_name": self.arguments.name,
                    "client_version": "0.1.0",
                    "resume_state": self.resume(),
                    "capabilities": {
                        "audio_profiles": list(PROFILES),
                        "sample_rates_hz": [44100, 48000],
                        "bit_depths": [16],
                        "max_channels": 2,
                        "can_pause": True,
                        "can_set_volume": True,
                        "can_mute": True,
                    },
                    "audio_profile": self.arguments.profile,
                },
            )
            LOGGER.debug("Hello sent: resume=%s", self.resume())
            async with asyncio.timeout(30):
                envelope = await self.receive(self.control, welcome=True)
            if envelope["type"] == "error":
                code = envelope["payload"]["error"]["code"]
                LOGGER.warning("Server rejected hello: %s", code)
                if code in (
                    "cache_unavailable",
                    "stale_playback",
                    "stream_unavailable",
                ):
                    await self.operate(self.output.stop)
                    raise ConnectionError("Server cannot resume retained playback")
                raise AudioError("Server rejected hello: " + str(code))
            if envelope["type"] != "welcome" or not isinstance(
                envelope["connection_epoch"], str
            ):
                raise AudioError("Expected a welcome with a connection epoch")
            self.epoch = envelope["connection_epoch"]
            welcome = envelope["payload"]
            self.reconcile(welcome)
            limits = welcome["limits"]
            if (
                limits["max_audio_chunk_payload_bytes"] != MAX_PAYLOAD
                or limits["max_unacknowledged_audio_chunks"] != 1
            ):
                raise AudioError("Incompatible server audio limits")
            LOGGER.info(
                "Connected: MA player %s, profile %s",
                welcome["player_id"],
                welcome["selected_audio_profile"],
            )
            tasks = [
                asyncio.create_task(self.control_loop()),
                asyncio.create_task(
                    self.audio_loop(session, welcome["audio_binding_id"], ws_timeout)
                ),
                asyncio.create_task(self.report_loop()),
                asyncio.create_task(self.ping_loop()),
            ]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            raise ConnectionError("WebSocket closed")
        finally:
            self.ready.clear()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            sockets = [
                connection
                for connection in (self.audio, self.control)
                if connection is not None
            ]
            self.audio = self.control = None
            await asyncio.gather(
                *(connection.close() for connection in sockets), return_exceptions=True
            )

    def reconcile(self, welcome: dict[str, Any]) -> None:
        """Require the server to acknowledge both retained slots and their cursors."""
        with self.output.lock:
            if welcome["media_revision"] != self.output.media_revision:
                raise ConnectionError("Media revision changed during reconnect")
            for state, stream in (
                (welcome["current_stream"], self.output.current),
                (welcome["prepared_next"], self.output.next),
            ):
                if state is None and stream is None:
                    continue
                if (
                    state is None
                    or stream is None
                    or state["stream"] != stream.descriptor
                ):
                    raise ConnectionError("Media slots changed during reconnect")
                received = state["next_received_chunk_index"]
                produced = state["next_produced_chunk_index"]
                if (
                    type(received) is not int
                    or type(produced) is not int
                    or received != stream.next_index
                    or not received <= produced <= 9007199254740991
                ):
                    raise ConnectionError("Retained audio cursor was not reconciled")

    async def receive(
        self, connection: aiohttp.ClientWebSocketResponse, *, welcome: bool = False
    ) -> dict[str, Any]:
        while True:
            message = await connection.receive()
            if message.type == aiohttp.WSMsgType.PING:
                await connection.pong(message.data)
            elif message.type == aiohttp.WSMsgType.PONG:
                if connection is self.control and secrets.compare_digest(
                    message.data, self.ping_payload
                ):
                    self.pong.set()
            elif message.type == aiohttp.WSMsgType.TEXT:
                try:
                    envelope = json.loads(message.data)
                except ValueError as error:
                    raise AudioError("Invalid server JSON") from error
                if (
                    not isinstance(envelope, dict)
                    or set(envelope)
                    != {
                        "protocol_version",
                        "type",
                        "message_id",
                        "connection_epoch",
                        "payload",
                    }
                    or type(envelope["protocol_version"]) is not int
                    or envelope["protocol_version"] != PROTOCOL_VERSION
                    or not isinstance(envelope["payload"], dict)
                    or (not welcome and envelope["connection_epoch"] != self.epoch)
                ):
                    raise AudioError("Invalid server envelope or stale epoch")
                LOGGER.log(
                    TRACE,
                    "Received %s on %s",
                    envelope["type"],
                    "Control" if connection is self.control else "Audio",
                )
                return envelope
            else:
                raise ConnectionError(
                    f"{'Control' if connection is self.control else 'Audio'} WebSocket "
                    f"frame={message.type.name}, close_code={connection.close_code}, "
                    f"error={connection.exception()}"
                )

    async def control_loop(self) -> None:
        assert self.control is not None
        while True:
            envelope = await self.receive(self.control)
            kind, payload = envelope["type"], envelope["payload"]
            if kind in COMMANDS:
                await self.command(kind, payload)
            elif kind == "error":
                LOGGER.warning("Server error: %s", payload["error"]["code"])
                with self.output.lock:
                    for stream in (self.output.current, self.output.next):
                        if (
                            stream is not None
                            and stream.identifier == payload["stream_id"]
                        ):
                            stream.error = payload["error"]
            else:
                raise AudioError("Unexpected Control message: " + str(kind))

    async def command(self, kind: str, payload: dict[str, Any]) -> None:
        async with self.output_lock:
            await self._command(kind, payload)

    async def _command(self, kind: str, payload: dict[str, Any]) -> None:
        identifier = payload["command_id"]
        previous = self.acks.get(identifier)
        if previous is not None:
            if previous[:2] != (kind, payload):
                raise AudioError("Command ID reused with different contents")
            await self.send("command_ack", previous[2])
            LOGGER.debug("Repeated command acknowledged: %s id=%s", kind, identifier)
            return
        LOGGER.info("MA command: %s id=%s", kind, identifier)
        error = None
        previous_activation = self.output.activation
        try:
            if kind == "set_media":
                if set(payload) != {
                    "command_id",
                    "media_revision",
                    "current",
                    "prepared",
                    "initial_output_state",
                }:
                    raise AudioError("Invalid complete media payload")
                await self.operate(
                    self.output.set_media,
                    payload["current"],
                    payload["prepared"],
                    payload["media_revision"],
                    payload["initial_output_state"],
                )
                LOGGER.info(
                    "Media registered: current=%s, prepared=%s, revision=%s",
                    payload["current"]["stream_id"] if payload["current"] else None,
                    payload["prepared"]["stream_id"] if payload["prepared"] else None,
                    payload["media_revision"],
                )
            elif kind in ("play", "pause", "stop"):
                current = self.output.current
                if payload["stream_id"] != (current.identifier if current else None):
                    raise AudioError("Command belongs to another stream")
                if kind == "stop":
                    await self.operate(self.output.stop)
                else:
                    await self.operate(self.output.set_playing, kind == "play")
            elif kind == "set_volume":
                value = payload["volume_percent"]
                if type(value) is not int or not 0 <= value <= 100:
                    raise AudioError("Invalid volume")
                with self.output.lock:
                    self.output.volume = value
            elif kind == "set_mute":
                if type(payload["is_muted"]) is not bool:
                    raise AudioError("Invalid mute state")
                with self.output.lock:
                    self.output.muted = payload["is_muted"]
        except (AudioError, sd.PortAudioError) as failure:
            LOGGER.error("Cannot apply %s: %s", kind, failure)
            error = protocol_error(
                "command_failed",
                "Playback command could not be applied",
            )
        ack = {
            "command_id": identifier,
            "status": "accepted" if error is None else "rejected",
            "error": error,
        }
        self.acks[identifier] = (kind, payload, ack)
        if len(self.acks) > 32:
            self.acks.popitem(last=False)
        await self.send("command_ack", ack)
        LOGGER.debug(
            "Command result: %s id=%s status=%s", kind, identifier, ack["status"]
        )
        if (
            error is None
            and self.output.activation is not None
            and self.output.activation != previous_activation
        ):
            receipt = self.output.activation
            assert receipt is not None
            snapshot = self.output.snapshot()
            await self.send(
                "prepared_activated",
                {
                    "activation": asdict(receipt),
                    "output_revision": snapshot["output_revision"],
                    "output_state": snapshot["output_state"],
                    "position_frame": snapshot["position_frame"],
                },
                sequence=True,
            )
        if self.ready.is_set():
            await self.capacity(force=True)
        await self.state()

    async def audio_loop(
        self,
        session: aiohttp.ClientSession,
        binding: str,
        timeout: aiohttp.ClientWSTimeout,
    ) -> None:
        LOGGER.debug("Connecting Audio channel")
        self.audio = await session.ws_connect(
            self.arguments.server + "/mashin/v1/audio",
            autoping=False,
            max_msg_size=HEADER.size + MAX_PAYLOAD,
            timeout=timeout,
        )
        await self.audio.send_json(
            {
                "protocol_version": PROTOCOL_VERSION,
                "type": "audio_hello",
                "message_id": str(uuid4()),
                "connection_epoch": self.epoch,
                "payload": {
                    "device_id": self.arguments.device_id,
                    "audio_binding_id": binding,
                    "audio_connection_id": self.audio_id,
                    "received_streams": self.output.cursors(),
                },
            }
        )
        async with asyncio.timeout(30):
            ready = await self.receive(self.audio)
        if ready["type"] != "audio_ready" or ready["payload"]:
            raise AudioError("Expected audio_ready")
        self.ready.set()
        self.disconnected_at = None
        LOGGER.info("Audio channel ready: connection_id=%s", self.audio_id)
        await self.capacity(force=True)
        while True:
            message = await self.audio.receive()
            if message.type == aiohttp.WSMsgType.PING:
                await self.audio.pong(message.data)
                continue
            if message.type == aiohttp.WSMsgType.PONG:
                continue
            if message.type != aiohttp.WSMsgType.BINARY:
                raise ConnectionError(
                    f"Audio WebSocket frame={message.type.name}, "
                    f"close_code={self.audio.close_code}, error={self.audio.exception()}"
                )
            data = message.data
            if not HEADER.size < len(data) <= HEADER.size + MAX_PAYLOAD:
                raise AudioError("Invalid audio frame")
            raw_id, index, _, _, _ = HEADER.unpack_from(data)
            stream_id = str(UUID(bytes=raw_id))
            with self.output.lock:
                stream = next(
                    (
                        item
                        for item in (self.output.current, self.output.next)
                        if item is not None and item.identifier == stream_id
                    ),
                    None,
                )
            if stream is None:
                LOGGER.log(
                    TRACE,
                    "Discarding chunk for retired stream: stream=%s index=%s",
                    stream_id,
                    index,
                )
                continue
            error = None
            try:
                chunk = self.output.accept(stream, data)
                LOGGER.log(
                    TRACE,
                    "Encoded chunk retained: stream=%s index=%s frames=%s final=%s bytes=%s buffered_bytes=%s",
                    stream_id,
                    index,
                    chunk.count,
                    chunk.final,
                    len(data),
                    self.output.buffered_bytes(),
                )
            except AudioError as failure:
                LOGGER.warning(
                    "Audio chunk rejected: stream=%s index=%s: %s",
                    stream_id,
                    index,
                    failure,
                    exc_info=self.arguments.debug,
                )
                error = protocol_error(
                    "stream_failed", "Audio chunk could not be validated or retained"
                )
            await self.send(
                "chunk_ack",
                {
                    "stream_id": stream_id,
                    "audio_connection_id": self.audio_id,
                    "chunk_index": index,
                    "next_chunk_index": stream.next_index,
                    "status": "accepted" if error is None else "rejected",
                    "buffer_capacity": self.output.capacity(),
                    "error": error,
                },
                sequence=True,
            )

    async def capacity(self, *, force: bool = False) -> None:
        value = self.output.capacity()
        if force or value != self.last_capacity:
            await self.send(
                "buffer_status",
                {"audio_connection_id": self.audio_id, "buffer_capacity": value},
                sequence=True,
            )
            self.last_capacity = value
            LOGGER.log(
                TRACE,
                "Buffer credit=%s payload bytes; encoded chunk budget=%s bytes; retained=%s bytes",
                value,
                self.output.buffer_capacity,
                self.output.buffered_bytes(),
            )

    async def state(self) -> None:
        snapshot = self.output.snapshot()
        await self.send("player_state", snapshot, sequence=True)
        identity = tuple(
            snapshot[key]
            for key in ("stream_id", "output_state", "volume_percent", "is_muted")
        )
        if identity != self.last_state:
            LOGGER.info(
                "Playback: state=%s stream=%s position=%s frames volume=%s mute=%s",
                snapshot["output_state"],
                snapshot["stream_id"],
                snapshot["position_frame"],
                snapshot["volume_percent"],
                snapshot["is_muted"],
            )
            self.last_state = identity
        else:
            LOGGER.log(
                TRACE,
                "Playback position: stream=%s frame=%s",
                snapshot["stream_id"],
                snapshot["position_frame"],
            )

    def log_buffer_summary(self, now: float) -> None:
        if now - self.last_buffer_summary < 5:
            return
        self.last_buffer_summary = now
        if LOGGER.isEnabledFor(logging.DEBUG):
            used, chunks, seconds = self.output.buffer_stats()
            LOGGER.debug(
                "Buffer: %.1f / %.1f KiB occupied, %s chunks cached, %.2f s remaining",
                used / 1024,
                self.output.buffer_capacity / 1024,
                chunks,
                seconds,
            )

    async def report_loop(self) -> None:
        last_report = 0.0
        while True:
            if self.output.failure:
                raise AudioError(self.output.failure)
            async with self.output_lock:
                events = await self.operate(self.output.poll)
                for kind, payload in events:
                    await self.send(kind, payload, sequence=True)
                    LOGGER.info(
                        "Playback event: %s stream=%s", kind, payload["stream_id"]
                    )
                if self.ready.is_set():
                    await self.capacity()
                if events or time.monotonic() - last_report >= 1:
                    await self.state()
                    last_report = time.monotonic()
            await asyncio.sleep(0.1)

    async def ping_loop(self) -> None:
        assert self.control is not None
        while True:
            await asyncio.sleep(5)
            self.ping_payload = secrets.token_bytes(16)
            self.pong.clear()
            async with asyncio.timeout(15):
                await self.control.ping(self.ping_payload)
                await self.pong.wait()
            LOGGER.log(TRACE, "Control ping acknowledged")


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a Mashin MA player on the local audio device; control playback in MA."
    )
    parser.add_argument(
        "server",
        nargs="?",
        type=server_url,
        help="MA base URL, e.g. http://192.168.1.10:8095",
    )
    parser.add_argument(
        "--device-id",
        default=str(uuid5(NAMESPACE_DNS, "mashin-py-cli:" + socket.gethostname())),
        help="Stable unique device ID; use different IDs for simultaneous clients",
    )
    parser.add_argument(
        "--name",
        default="Mashin " + socket.gethostname(),
        help="Player name shown in MA",
    )
    parser.add_argument(
        "--codec",
        "--profile",
        dest="profile",
        choices=("opus", *PROFILES),
        default="flac",
        help="Codec/profile (default: flac; opus selects opus_128k); --profile is an alias",
    )
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=4,
        metavar="MiB",
        help="Encoded server-chunk buffer budget in MiB, minimum 1 (default: 4)",
    )
    parser.add_argument(
        "--output-device",
        help="SoundDevice output index or device-name substring (default: system default)",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List local audio devices and exit"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging with a buffer summary every five seconds",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Include per-chunk and protocol logging (implies --debug)",
    )
    result = parser.parse_args(argv)
    result.debug = result.debug or result.trace
    if not result.server and not result.list_devices:
        parser.error("a server URL is required unless --list-devices is used")
    if result.buffer_capacity < 1:
        parser.error("--buffer-capacity must be at least 1 MiB")
    if result.profile == "opus":
        result.profile = "opus_128k"
    if (
        not result.device_id
        or not result.device_id.isascii()
        or len(result.device_id) > 128
    ):
        parser.error("--device-id must be 1..128 ASCII characters")
    if result.output_device is not None and result.output_device.isdecimal():
        result.output_device = int(result.output_device)
    return result


def main() -> None:
    options = arguments()
    logging.basicConfig(
        level=TRACE
        if options.trace
        else logging.DEBUG
        if options.debug
        else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if options.list_devices:
        print(sd.query_devices())
        return
    try:
        asyncio.run(Client(options).run())
    except KeyboardInterrupt:
        pass
    except (AudioError, sd.PortAudioError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
