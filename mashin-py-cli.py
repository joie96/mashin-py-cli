# /// script
# requires-python = ">=3.12"
# dependencies = ["aiohttp>=3.12,<4", "av>=19,<20", "numpy>=2,<3", "sounddevice>=0.5.6,<0.6"]
# ///
"""Mashin V1 console receiver. Playback is controlled exclusively through MA."""

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
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

import aiohttp
import sounddevice as sd
from mashin_audio import (
    HEADER,
    MAX_PAYLOAD,
    PROFILES,
    AudioError,
    Output,
    decode_chunk,
    stream_from,
)

LOGGER = logging.getLogger("mashin")
COMMANDS = {
    "set_media",
    "prepare_next",
    "clear_next",
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
        self.output = output or Output(arguments.output_device)
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
                    "protocol_version": 1,
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
            "next_revision": self.output.next_revision,
            "received_streams": self.output.cursors(),
        }

    async def run(self) -> None:
        delay = 0.5
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
        watchdog = asyncio.create_task(self.grace_loop())
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
                            "Connection lost (%s); reconnecting in %.1f s",
                            type(error).__name__,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        delay = min(8, delay * 2)
        finally:
            watchdog.cancel()
            await asyncio.gather(watchdog, return_exceptions=True)
            await self.operate(self.output.stop)

    async def grace_loop(self) -> None:
        while True:
            await asyncio.sleep(0.25)
            if (
                self.disconnected_at is not None
                and time.monotonic() - self.disconnected_at >= 30
            ):
                LOGGER.warning("Reconnect grace expired; stopping retained playback")
                async with self.output_lock:
                    await self.operate(self.output.stop)
                self.disconnected_at = None

    async def connect(self, session: aiohttp.ClientSession) -> None:
        await self.operate(self.output.poll)
        self.epoch, self.sequence = None, 0
        self.acks.clear()
        self.ready.clear()
        self.last_capacity = None
        tasks: list[asyncio.Task[None]] = []
        ws_timeout = aiohttp.ClientWSTimeout(ws_close=2)
        try:
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
            async with asyncio.timeout(30):
                envelope = await self.receive(self.control, welcome=True)
            if envelope["type"] == "error":
                code = envelope["payload"]["error"]["code"]
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
            snapshot = self.output.snapshot()
            accepted = welcome["current_stream"]
            if (accepted["stream"]["stream_id"] if accepted else None) != snapshot[
                "stream_id"
            ]:
                raise ConnectionError("Playback changed during reconnect")
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
                    or envelope["protocol_version"] != 1
                    or not isinstance(envelope["payload"], dict)
                    or (not welcome and envelope["connection_epoch"] != self.epoch)
                ):
                    raise AudioError("Invalid server envelope or stale epoch")
                return envelope
            else:
                raise ConnectionError("WebSocket closed or unexpected frame")

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
            return
        error = None
        try:
            if kind == "set_media":
                await self.operate(
                    self.output.load, payload["stream"], payload["autoplay"]
                )
            elif kind == "prepare_next":
                stream = stream_from(payload["stream"])
                with self.output.lock:
                    if (
                        self.output.current is None
                        or self.output.current.identifier
                        != payload["current_stream_id"]
                    ):
                        raise AudioError(
                            "Successor belongs to a different current stream"
                        )
                    if (
                        self.output.next is not None
                        or payload["next_revision"] <= self.output.next_revision
                    ):
                        raise AudioError(
                            "A successor is already prepared or its revision is stale"
                        )
                    self.output.next = stream
                    self.output.next_revision = payload["next_revision"]
            elif kind == "clear_next":
                with self.output.lock:
                    if (
                        self.output.current is None
                        or self.output.current.identifier
                        != payload["current_stream_id"]
                    ):
                        raise AudioError(
                            "Successor already promoted or current stream changed"
                        )
                    self.output.next = None
                    self.output.next_revision = payload["next_revision"]
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
                "stale_playback" if kind == "clear_next" else "command_failed",
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
        if self.ready.is_set():
            await self.capacity(force=True)
        await self.state()

    async def audio_loop(
        self,
        session: aiohttp.ClientSession,
        binding: str,
        timeout: aiohttp.ClientWSTimeout,
    ) -> None:
        self.audio = await session.ws_connect(
            self.arguments.server + "/mashin/v1/audio",
            autoping=False,
            max_msg_size=HEADER.size + MAX_PAYLOAD,
            timeout=timeout,
        )
        await self.audio.send_json(
            {
                "protocol_version": 1,
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
        await self.capacity(force=True)
        while True:
            message = await self.audio.receive()
            if message.type == aiohttp.WSMsgType.PING:
                await self.audio.pong(message.data)
                continue
            if message.type == aiohttp.WSMsgType.PONG:
                continue
            if message.type != aiohttp.WSMsgType.BINARY:
                raise ConnectionError("Audio WebSocket closed")
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
                continue
            error = None
            try:
                chunk = await asyncio.to_thread(decode_chunk, data, stream)
                self.output.accept(stream, chunk)
            except AudioError:
                error = protocol_error(
                    "stream_failed", "Audio chunk could not be decoded or retained"
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

    async def state(self) -> None:
        await self.send("player_state", self.output.snapshot(), sequence=True)

    async def report_loop(self) -> None:
        last_report = 0.0
        while True:
            if self.output.failure:
                raise AudioError(self.output.failure)
            async with self.output_lock:
                events = await self.operate(self.output.poll)
                for kind, payload in events:
                    await self.send(kind, payload, sequence=True)
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
        "--profile",
        choices=PROFILES,
        default="flac",
        help="Audio profile (default: flac)",
    )
    parser.add_argument(
        "--output-device",
        help="SoundDevice output index or device-name substring (default: system default)",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List local audio devices and exit"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable diagnostic logging"
    )
    result = parser.parse_args(argv)
    if not result.server and not result.list_devices:
        parser.error("a server URL is required unless --list-devices is used")
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
        level=logging.DEBUG if options.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
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
