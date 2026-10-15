"""Synthetic codec and audio-clock checks; no server or sound card required."""

import asyncio
import importlib.util
import io
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import aiohttp
import av
import numpy as np
import sounddevice as sd
from aiohttp import web
from aiohttp.test_utils import TestServer
from mashin_audio import (
    HEADER,
    MAX_PCM,
    PROFILES,
    AudioError,
    Chunk,
    Output,
    decode_chunk,
    stream_from,
)

SPEC = importlib.util.spec_from_file_location(
    "mashin_client", Path(__file__).with_name("mashin-py-cli.py")
)
client_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client_module)


def descriptor(profile="flac"):
    result = {
        "stream_id": str(uuid4()),
        "output_format": {
            "profile_id": "flac",
            "container": "flac",
            "codec": "flac",
            "sample_rate_hz": 48000,
            "bit_depth": 16,
            "channels": 2,
            "target_bitrate_bps": None,
            "uses_vbr": False,
        },
    }
    if profile != "flac":
        result["output_format"].update(
            profile_id=profile,
            container="ogg",
            codec="opus",
            bit_depth=None,
            target_bitrate_bps=int(profile[5:-1]) * 1000,
            uses_vbr=True,
        )
    return result


def flac_chunk(stream, count=48000, final=True):
    target = io.BytesIO()
    opus = stream.descriptor["output_format"]["codec"] == "opus"
    with av.open(target, mode="w", format="ogg" if opus else "flac") as container:
        encoder = container.add_stream("libopus" if opus else "flac", rate=48000)
        if opus:
            encoder.bit_rate = stream.descriptor["output_format"]["target_bitrate_bps"]
        encoder.layout = "stereo"
        frame = av.AudioFrame.from_ndarray(
            np.zeros((1, count * 2), dtype=np.int16), format="s16", layout="stereo"
        )
        frame.sample_rate = 48000
        frame.pts = 0
        frame.time_base = Fraction(1, 48000)
        for packet in encoder.encode(frame):
            container.mux(packet)
        for packet in encoder.encode(None):
            container.mux(packet)
    identifier = UUID(stream.identifier)
    return HEADER.pack(identifier.bytes, 1, 0, count, int(final)) + target.getvalue()


class AudioTests(unittest.TestCase):
    def test_all_opus_profiles_apply_preskip_and_final_trim(self):
        for profile in PROFILES[1:]:
            stream = stream_from(descriptor(profile))
            for count in (1, 47999, 48000):
                with self.subTest(profile=profile, count=count):
                    self.assertEqual(
                        decode_chunk(flac_chunk(stream, count), stream).pcm.shape,
                        (count, 2),
                    )

    def test_pcm_capacity_is_bounded(self):
        output = Output()
        output.current = stream = stream_from(descriptor())
        while output.capacity():
            chunk = Chunk(
                stream.identifier,
                stream.next_index,
                stream.received_frames,
                48000,
                False,
                np.zeros((48000, 2), np.float32),
            )
            output.accept(stream, chunk)
        self.assertLessEqual(sum(chunk.pcm.nbytes for chunk in stream.chunks), MAX_PCM)
        with self.assertRaises(AudioError):
            output.accept(stream, chunk)

    def test_decode_final_flac_and_frame_count(self):
        stream = stream_from(descriptor())
        for count in (1, 47999, 48000):
            with self.subTest(count=count):
                chunk = decode_chunk(flac_chunk(stream, count), stream)
                self.assertEqual(chunk.pcm.shape, (count, 2))

    def test_reject_incorrect_frame_count(self):
        stream = stream_from(descriptor())
        data = flac_chunk(stream)
        with self.assertRaises(AudioError):
            decode_chunk(data[:32] + (1).to_bytes(8, "big") + data[40:], stream)

    def test_track_ends_at_dac_not_on_receipt(self):
        stream = stream_from(descriptor())
        output = Output()
        output.current = stream
        output.playing = True
        output.output = Mock(time=10.0)
        output.accept(stream, decode_chunk(flac_chunk(stream), stream))
        block = np.empty((48000, 2), dtype=np.float32)
        output._callback(block, 48000, SimpleNamespace(outputBufferDacTime=11.0), None)
        self.assertEqual(output.poll(), [])
        self.assertEqual(stream.position, 0)
        output.output.time = 11.5
        self.assertEqual(output.poll(), [])
        self.assertEqual(stream.position, 24000)
        output.output.time = 12.0
        self.assertEqual(
            output.poll(), [("track_ended", {"stream_id": stream.identifier})]
        )
        self.assertEqual(output.poll(), [])


class FakeOutput(Output):
    def _open(self, stream):
        self.close_device()
        self.output = Mock(time=0.0, active=True)


class CommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_pause_accounts_for_drained_frames_with_lagging_device_clock(self):
        output = FakeOutput()
        output.load(descriptor(), True)
        stream = output.current
        output.accept(stream, decode_chunk(flac_chunk(stream), stream))
        output._callback(
            np.empty((48000, 2), np.float32),
            48000,
            SimpleNamespace(outputBufferDacTime=1.0),
            None,
        )
        output.set_playing(False)
        self.assertEqual(output.snapshot()["position_frame"], 48000)
        self.assertEqual(output.snapshot()["output_state"], "paused")
        self.assertFalse(output.scheduled)

    async def test_failed_device_open_rejects_load_and_releases_stream(self):
        output = FakeOutput()
        output._open = Mock(side_effect=sd.PortAudioError("No output available"))
        client = client_module.Client(
            client_module.arguments(["localhost:8095"]), output
        )
        client.send = AsyncMock()
        with self.assertLogs("mashin", level="ERROR"):
            await client.command(
                "set_media", {"command_id": 1, "stream": descriptor(), "autoplay": True}
            )
        self.assertEqual(client.send.await_args_list[0].args[1]["status"], "rejected")
        self.assertIsNone(output.current)
        self.assertEqual(output.snapshot()["output_state"], "idle")

    async def test_commands_and_duplicate_load(self):
        output = FakeOutput()
        client = client_module.Client(
            client_module.arguments(["localhost:8095"]), output
        )
        client.send = AsyncMock()
        client.ready.set()
        media = {"command_id": 1, "stream": descriptor(), "autoplay": True}
        await client.command("set_media", media)
        current = output.current
        self.assertEqual(
            [call.args[0] for call in client.send.await_args_list],
            ["command_ack", "buffer_status", "player_state"],
        )
        await client.command("set_media", media)
        self.assertIs(output.current, current)
        for identifier, kind, values in (
            (2, "pause", {"stream_id": current.identifier}),
            (3, "set_volume", {"volume_percent": 45}),
            (4, "set_mute", {"is_muted": True}),
        ):
            await client.command(kind, {"command_id": identifier, **values})
        self.assertEqual(output.snapshot()["output_state"], "paused")
        self.assertEqual(output.volume, 45)
        self.assertTrue(output.muted)
        await client.command("play", {"command_id": 5, "stream_id": current.identifier})
        self.assertTrue(output.playing)
        await client.command("stop", {"command_id": 6, "stream_id": current.identifier})
        self.assertEqual(output.snapshot()["output_state"], "idle")

    async def test_promotion_orders_ended_before_started(self):
        output = FakeOutput()
        output.load(descriptor(), True)
        current = output.current
        output.next = following = stream_from(descriptor())
        output.next_revision = 3
        for stream in (current, following):
            output.accept(stream, decode_chunk(flac_chunk(stream), stream))
        output._callback(
            np.empty((48000, 2), np.float32),
            48000,
            SimpleNamespace(outputBufferDacTime=1.0),
            None,
        )
        output.output.time = 2.0
        self.assertEqual(
            [kind for kind, _ in output.poll()], ["track_ended", "track_started"]
        )
        self.assertIs(output.current, following)
        self.assertEqual(output.snapshot()["position_frame"], 0)


class NetworkTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.output = FakeOutput()
        self.messages = asyncio.Queue()
        self.audio_open = asyncio.Event()
        self.hellos = []
        self.epoch = str(uuid4())
        app = web.Application()
        app.router.add_get("/mashin/v1/control", self.control_handler)
        app.router.add_get("/mashin/v1/audio", self.audio_handler)
        self.server = TestServer(app)
        await self.server.start_server()
        self.client = client_module.Client(
            client_module.arguments([str(self.server.make_url("/"))]),
            self.output,
        )
        self.session = aiohttp.ClientSession()
        self.task = asyncio.create_task(self.client.connect(self.session))
        await asyncio.wait_for(self.audio_open.wait(), 5)

    async def asyncTearDown(self):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        await self.session.close()
        await self.server.close()

    def envelope(self, kind, payload):
        return {
            "protocol_version": 1,
            "type": kind,
            "message_id": str(uuid4()),
            "connection_epoch": self.epoch,
            "payload": payload,
        }

    async def control_handler(self, request):
        connection = web.WebSocketResponse()
        await connection.prepare(request)
        hello = await connection.receive_json()
        self.hellos.append(hello)
        self.control = connection
        stream = self.output.current
        await connection.send_json(
            self.envelope(
                "welcome",
                {
                    "player_id": "test-player",
                    "selected_audio_profile": "flac",
                    "audio_binding_id": "test-binding",
                    "current_stream": {"stream": stream.descriptor} if stream else None,
                    "limits": {
                        "max_audio_chunk_payload_bytes": 262144,
                        "max_unacknowledged_audio_chunks": 1,
                    },
                },
            )
        )
        async for message in connection:
            if message.type == aiohttp.WSMsgType.TEXT:
                await self.messages.put(message.json())
        return connection

    async def audio_handler(self, request):
        connection = web.WebSocketResponse()
        await connection.prepare(request)
        self.audio_hello = await connection.receive_json()
        self.audio = connection
        await connection.send_json(self.envelope("audio_ready", {}))
        self.audio_open.set()
        async for _ in connection:
            pass
        return connection

    async def until(self, kind):
        async with asyncio.timeout(5):
            while True:
                message = await self.messages.get()
                if message["type"] == kind:
                    return message

    async def load(self, autoplay=True):
        await self.until("buffer_status")
        media = descriptor()
        await self.control.send_json(
            self.envelope(
                "set_media",
                {
                    "command_id": 1,
                    "stream": media,
                    "autoplay": autoplay,
                },
            )
        )
        ack = await self.until("command_ack")
        self.assertEqual(ack["payload"]["status"], "accepted")
        capacity = await self.until("buffer_status")
        self.assertGreater(capacity["payload"]["buffer_capacity"], 0)
        self.assertGreater(capacity["payload"]["sequence"], 0)
        await self.audio.send_bytes(flac_chunk(self.output.current))
        chunk_ack = await self.until("chunk_ack")
        self.assertEqual(chunk_ack["payload"]["status"], "accepted")
        self.assertEqual(chunk_ack["payload"]["next_chunk_index"], 2)
        return media

    async def test_handshake_load_audio_and_audible_end(self):
        media = await self.load()
        self.assertEqual(
            self.hellos[0]["payload"]["capabilities"]["audio_profiles"], list(PROFILES)
        )
        self.assertEqual(self.audio_hello["connection_epoch"], self.epoch)
        self.output._callback(
            np.empty((48000, 2), np.float32),
            48000,
            SimpleNamespace(outputBufferDacTime=1.0),
            None,
        )
        self.assertFalse(self.output.current.ended)
        self.output.output.time = 2.0
        ended = await self.until("track_ended")
        self.assertEqual(ended["payload"]["stream_id"], media["stream_id"])
        state = await self.until("player_state")
        self.assertEqual(state["payload"]["position_frame"], 48000)

    async def test_reconnect_retains_receipt_cursor_without_replaying(self):
        media = await self.load(autoplay=False)
        current = self.output.current
        await self.client.control.close()
        with self.assertRaises(ConnectionError):
            await asyncio.wait_for(self.task, 5)
        self.audio_open.clear()
        self.epoch = str(uuid4())
        self.task = asyncio.create_task(self.client.connect(self.session))
        await asyncio.wait_for(self.audio_open.wait(), 5)
        await self.until("buffer_status")
        self.assertIs(self.output.current, current)
        self.assertEqual(
            self.hellos[-1]["payload"]["resume_state"]["received_streams"],
            [{"stream_id": media["stream_id"], "next_chunk_index": 2}],
        )
        self.assertEqual(
            self.audio_hello["payload"]["received_streams"], self.output.cursors()
        )
        self.assertEqual(self.output.snapshot()["output_state"], "paused")


if __name__ == "__main__":
    unittest.main()
