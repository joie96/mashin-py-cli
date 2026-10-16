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
    DEFAULT_BUFFER_CAPACITY,
    HEADER,
    MAX_PAYLOAD,
    PROFILES,
    AudioError,
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
    return (
        HEADER.pack(
            identifier.bytes,
            stream.next_index,
            stream.received_frames,
            count,
            int(final),
        )
        + target.getvalue()
    )


class AudioTests(unittest.TestCase):
    def test_manual_activation_retains_cache_and_rejects_stale_grants(self):
        output = Output()
        current = stream_from(descriptor())
        successor = stream_from(descriptor())
        output.current, output.next, output.next_revision = current, successor, 1
        data = flac_chunk(successor)
        output.accept(successor, data)
        cache = successor.chunks
        cursor = successor.next_index
        with unittest.mock.patch.object(output, "_open"):
            with self.assertRaises(AudioError):
                output.activate_prepared(current.identifier, successor.identifier, 2)
            self.assertIs(output.current, current)
            output.activate_prepared(current.identifier, successor.identifier, 1)
            self.assertIs(output.current, successor)
            self.assertIs(successor.chunks, cache)
            self.assertEqual(successor.next_index, cursor)
            self.assertIsNone(output.next)
            self.assertEqual(output.activation.previous_stream_id, current.identifier)
            successor.position = 100
            output.activate_prepared(current.identifier, successor.identifier, 1)
            self.assertEqual(successor.position, 100)
        output.stop()
        self.assertIsNone(output.activation)

    def test_decoder_worker_refills_and_releases_wire_bytes_then_stops(self):
        output = Output()
        device = Mock(time=0.0, active=True, device=0)
        with (
            unittest.mock.patch("mashin_audio.sd.check_output_settings"),
            unittest.mock.patch("mashin_audio.sd.OutputStream", return_value=device),
            unittest.mock.patch(
                "mashin_audio.sd.query_devices", return_value={"name": "Test output"}
            ),
        ):
            try:
                output.load(descriptor(), False)
                stream = output.current
                for index in range(4):
                    output.accept(stream, flac_chunk(stream, final=index == 3))
                self.assertFalse(stream.decoded)
                retained = output.buffered_bytes()
                first_size = len(stream.chunks[0].data)
                output.set_playing(True)
                with output.condition:
                    self.assertTrue(output.condition.wait_for(lambda: len(stream.decoded) == 2, 3))
                worker = output.decoder
                output._callback(
                    np.empty((48000, 2), np.float32),
                    48000,
                    SimpleNamespace(outputBufferDacTime=1.0),
                    None,
                )
                with output.condition:
                    self.assertTrue(output.condition.wait_for(lambda: stream.decoded_index == 3, 3))
                self.assertEqual(output.buffered_bytes(), retained - first_size)
                self.assertEqual(len(stream.decoded), 2)
                self.assertEqual(stream.next_index, 5)
            finally:
                output.stop()
        self.assertFalse(worker.is_alive())
        self.assertIsNone(output.decoder)
        self.assertEqual(output.buffered_bytes(), 0)

    def test_deferred_decode_failure_reports_error_not_natural_end(self):
        output = Output()
        output.current = stream = stream_from(descriptor())
        output.output = Mock(time=2.0)
        output.playing = True
        data = HEADER.pack(UUID(stream.identifier).bytes, 1, 0, 48000, 1) + b"invalid codec data"
        output.accept(stream, data)
        self.assertEqual(stream.next_index, 2)
        with self.assertLogs("mashin.audio", level="ERROR"):
            self.assertFalse(output.prepare())
        self.assertEqual(output.snapshot()["output_state"], "error")
        self.assertEqual(output.snapshot()["error"]["code"], "stream_failed")
        self.assertEqual(output.poll(), [])

    def test_buffered_chunks_can_decode_after_a_server_source_error(self):
        output = Output()
        output.current = stream = stream_from(descriptor())
        output.accept(stream, flac_chunk(stream))
        stream.error = client_module.protocol_error("stream_failed", "Source ended with an error")
        output.playing = True
        self.assertTrue(output.prepare())
        self.assertEqual(len(stream.decoded), 1)

    def test_capacity_counts_original_encoded_chunks_without_decoding(self):
        size = HEADER.size + MAX_PAYLOAD
        output = Output(buffer_capacity=2 * size)
        output.current = stream = stream_from(descriptor())
        with unittest.mock.patch(
            "mashin_audio.decode_chunk",
            side_effect=AssertionError("Receipt must not decode"),
        ):
            for _ in range(2):
                data = (
                    HEADER.pack(
                        UUID(stream.identifier).bytes,
                        stream.next_index,
                        stream.received_frames,
                        48000,
                        0,
                    )
                    + b"x" * MAX_PAYLOAD
                )
                retained = output.accept(stream, data)
                self.assertIs(retained.data, data)
                self.assertFalse(hasattr(retained, "pcm"))
        self.assertEqual(output.buffered_bytes(), 2 * size)
        self.assertFalse(stream.decoded)
        self.assertEqual(stream.next_index, 3)
        self.assertEqual(output.capacity(), 0)
        with self.assertRaises(AudioError):
            Output(buffer_capacity=size - 1)

    def test_decode_working_set_is_independent_of_chunk_buffer(self):
        output = Output()
        output.current = stream = stream_from(descriptor("opus_128k"))
        for index in range(8):
            output.accept(stream, flac_chunk(stream, final=index == 7))
        self.assertFalse(output.prepare())
        self.assertFalse(stream.decoded)
        output.playing = True
        self.assertTrue(output.prepare())
        self.assertTrue(output.prepare())
        self.assertFalse(output.prepare())
        self.assertEqual(len(stream.decoded), 2)
        self.assertEqual(len(stream.chunks), 8)
        self.assertEqual(stream.next_index, 9)

    def test_all_opus_profiles_apply_preskip_and_final_trim(self):
        for profile in PROFILES[1:]:
            stream = stream_from(descriptor(profile))
            for count in (1, 47999, 48000):
                with self.subTest(profile=profile, count=count):
                    self.assertEqual(
                        decode_chunk(flac_chunk(stream, count), stream).pcm.shape,
                        (count, 2),
                    )

    def test_encoded_capacity_is_bounded_across_current_and_next(self):
        output = Output()
        output.current = stream = stream_from(descriptor())
        output.next = following = stream_from(descriptor())
        while output.capacity() == MAX_PAYLOAD:
            stream = following if stream is output.current else output.current
            data = (
                HEADER.pack(
                    UUID(stream.identifier).bytes,
                    stream.next_index,
                    stream.received_frames,
                    48000,
                    0,
                )
                + b"x" * MAX_PAYLOAD
            )
            output.accept(stream, data)
        self.assertLessEqual(output.buffered_bytes(), DEFAULT_BUFFER_CAPACITY)
        with self.assertRaises(AudioError):
            data = (
                HEADER.pack(
                    UUID(stream.identifier).bytes,
                    stream.next_index,
                    stream.received_frames,
                    48000,
                    0,
                )
                + b"x" * MAX_PAYLOAD
            )
            output.accept(stream, data)

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
        output.accept(stream, flac_chunk(stream))
        output.prepare()
        block = np.empty((48000, 2), dtype=np.float32)
        output._callback(block, 48000, SimpleNamespace(outputBufferDacTime=11.0), None)
        self.assertEqual(output.poll(), [])
        self.assertEqual(stream.position, 0)
        output.output.time = 11.5
        self.assertEqual(output.poll(), [])
        self.assertEqual(stream.position, 24000)
        output.output.time = 12.0
        self.assertEqual(output.poll(), [("track_ended", {"stream_id": stream.identifier})])
        self.assertEqual(output.poll(), [])


class ArgumentTests(unittest.TestCase):
    def test_explicit_debug_and_trace_flags(self):
        normal = client_module.arguments(["localhost:8095"])
        self.assertFalse(normal.debug)
        self.assertFalse(normal.trace)
        debug = client_module.arguments(["localhost:8095", "--debug"])
        self.assertTrue(debug.debug)
        self.assertFalse(debug.trace)
        trace = client_module.arguments(["localhost:8095", "--trace"])
        self.assertTrue(trace.debug)
        self.assertTrue(trace.trace)
        with (
            self.assertRaises(SystemExit),
            unittest.mock.patch("sys.stderr", io.StringIO()),
        ):
            client_module.arguments(["localhost:8095", "--verbose"])

    def test_codec_alias_and_encoded_budget(self):
        options = client_module.arguments(
            ["localhost:8095", "--codec", "opus", "--buffer-capacity", "8"]
        )
        self.assertEqual(options.profile, "opus_128k")
        self.assertEqual(client_module.Client(options).output.buffer_capacity, 8 * 1024 * 1024)
        for profile in PROFILES:
            with self.subTest(profile=profile):
                self.assertEqual(
                    client_module.arguments(["localhost:8095", "--codec", profile]).profile,
                    profile,
                )
                self.assertEqual(
                    client_module.arguments(["localhost:8095", "--profile", profile]).profile,
                    profile,
                )

    def test_invalid_buffer_capacity(self):
        for value in ("0", "-1", "1.5", "invalid"):
            with (
                self.subTest(value=value),
                self.assertRaises(SystemExit),
                unittest.mock.patch("sys.stderr", io.StringIO()),
            ):
                client_module.arguments(["localhost:8095", "--buffer-capacity", value])


class FakeOutput(Output):
    def _open(self, stream):
        self.close_device()
        self.output = Mock(time=0.0, active=True)


class CommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_connection_failures_log_without_tracebacks(self):
        for flag in ("--debug", "--trace"):
            with self.subTest(flag=flag):
                client = client_module.Client(
                    client_module.arguments(["localhost:8095", flag]), FakeOutput()
                )
                client.connect = AsyncMock(
                    side_effect=[
                        aiohttp.ClientConnectionError("Network unreachable"),
                        TimeoutError(),
                        asyncio.CancelledError(),
                    ]
                )
                client.buffer_loop = AsyncMock()
                with (
                    unittest.mock.patch.object(
                        client_module.asyncio, "sleep", new_callable=AsyncMock
                    ),
                    self.assertLogs("mashin", level="WARNING") as captured,
                    self.assertRaises(asyncio.CancelledError),
                ):
                    await client.run()
                self.assertEqual(len(captured.records), 2)
                self.assertTrue(all(record.exc_info is None for record in captured.records))
                self.assertIn("reconnecting in 0.5 s", captured.output[0])
                self.assertIn("reconnecting in 1.0 s", captured.output[1])

    async def test_disconnected_buffer_keeps_playing_after_thirty_seconds(self):
        output = FakeOutput()
        output.load(descriptor(), True)
        stream = output.current
        output.accept(stream, flac_chunk(stream, final=False))
        output.accept(stream, flac_chunk(stream))
        output.prepare()
        client = client_module.Client(
            client_module.arguments(["localhost:8095", "--debug"]), output
        )
        client.disconnected_at = 0
        client.last_buffer_summary = 0
        client.operate = AsyncMock(side_effect=lambda operation, *args: operation(*args))
        clock = Mock(return_value=0)

        async def tick(_delay):
            if clock.return_value >= 40:
                raise asyncio.CancelledError()
            if clock.return_value == 35:
                output._callback(
                    np.empty((48000, 2), np.float32),
                    48000,
                    SimpleNamespace(outputBufferDacTime=1.0),
                    None,
                )
            clock.return_value = 35 if clock.return_value == 0 else 40

        with (
            unittest.mock.patch.object(client_module.asyncio, "sleep", side_effect=tick),
            unittest.mock.patch.object(client_module.time, "monotonic", clock),
            self.assertLogs("mashin", level="DEBUG") as captured,
            self.assertRaises(asyncio.CancelledError),
        ):
            await client.buffer_loop()
        self.assertIsNone(client.control)
        self.assertIs(output.current, stream)
        self.assertTrue(output.playing)
        self.assertEqual(client.disconnected_at, 0)
        self.assertEqual(len(captured.output), 2)
        self.assertIn("2 chunks cached, 2.00 s remaining", captured.output[0])
        self.assertIn("1 chunks cached, 1.00 s remaining", captured.output[1])

    async def test_reconnect_backoff_stays_at_eight_seconds(self):
        client = client_module.Client(client_module.arguments(["localhost:8095"]), FakeOutput())
        client.connect = AsyncMock(
            side_effect=[ConnectionError("Offline") for _ in range(9)] + [asyncio.CancelledError()]
        )
        client.buffer_loop = AsyncMock()
        with (
            unittest.mock.patch.object(
                client_module.asyncio, "sleep", new_callable=AsyncMock
            ) as sleep,
            self.assertLogs("mashin", level="WARNING"),
            self.assertRaises(asyncio.CancelledError),
        ):
            await client.run()
        delays = [call.args[0] for call in sleep.await_args_list]
        self.assertEqual(delays, [0.5, 1, 2, 4, 8, 8, 8, 8, 8])
        self.assertGreater(sum(delays), 30)

    async def test_offline_buffer_drains_current_and_authorized_successor(self):
        output = FakeOutput()
        output.load(descriptor(), True)
        current = output.current
        output.next = following = stream_from(descriptor())
        output.next_revision = 1
        for stream in (current, following):
            output.accept(stream, flac_chunk(stream))
        client = client_module.Client(client_module.arguments(["localhost:8095"]), output)
        client.disconnected_at = 0
        ticks = 0

        async def tick(_delay):
            nonlocal ticks
            if ticks == 2:
                raise asyncio.CancelledError()
            output.prepare()
            output._callback(
                np.empty((48000, 2), np.float32),
                48000,
                SimpleNamespace(outputBufferDacTime=float(ticks)),
                None,
            )
            ticks += 1
            output.output.time = float(ticks)

        with (
            unittest.mock.patch.object(client_module.asyncio, "sleep", side_effect=tick),
            self.assertLogs("mashin", level="INFO") as captured,
            self.assertRaises(asyncio.CancelledError),
        ):
            await client.buffer_loop()
        self.assertIs(output.current, following)
        self.assertIsNone(output.next)
        self.assertTrue(following.ended)
        self.assertEqual(following.position, 48000)
        self.assertEqual(output.buffered_bytes(), 0)
        messages = "\n".join(captured.output)
        self.assertIn("Offline playback event: track_started", messages)
        self.assertEqual(messages.count("Offline playback event: track_ended"), 2)

    async def test_command_and_state_logs(self):
        client = client_module.Client(client_module.arguments(["localhost:8095"]), FakeOutput())
        client.send = AsyncMock()
        with self.assertLogs("mashin", level="DEBUG") as captured:
            await client.command(
                "set_media", {"command_id": 1, "stream": descriptor(), "autoplay": True}
            )
            await client.state()
        messages = "\n".join(captured.output)
        self.assertIn("MA command: set_media id=1", messages)
        self.assertIn("Media loaded: stream=", messages)
        self.assertIn("Playback: state=buffering", messages)
        self.assertNotIn("Playback position:", messages)

    async def test_debug_buffer_summary_is_limited_to_five_seconds(self):
        output = FakeOutput()
        output.load(descriptor(), True)
        current = output.current
        output.accept(current, flac_chunk(current, final=False))
        output.accept(current, flac_chunk(current, count=24000))
        output.next = following = stream_from(descriptor())
        following.descriptor["output_format"]["sample_rate_hz"] = 44100
        data = HEADER.pack(UUID(following.identifier).bytes, 1, 0, 22050, 1) + b"encoded fixture"
        output.accept(following, data)
        output.prepare()
        output._callback(
            np.empty((24000, 2), np.float32),
            24000,
            SimpleNamespace(outputBufferDacTime=1.0),
            None,
        )
        client = client_module.Client(client_module.arguments(["localhost:8095"]), output)
        client.last_buffer_summary = 0
        with self.assertLogs("mashin", level="DEBUG") as captured:
            for now in (0, 4.99, 5, 5.1, 9.99, 10):
                client.log_buffer_summary(now)
        self.assertEqual(len(captured.output), 2)
        used, count, seconds = output.buffer_stats()
        self.assertEqual(count, 3)
        self.assertEqual(seconds, 1.5)
        self.assertIn(
            f"{used / 1024:.1f} / 4096.0 KiB occupied, 3 chunks cached, 1.50 s remaining",
            captured.output[0],
        )

    async def test_trace_retains_position_details(self):
        client = client_module.Client(
            client_module.arguments(["localhost:8095", "--trace"]), FakeOutput()
        )
        self.assertTrue(client.arguments.debug)
        client.send = AsyncMock()
        with self.assertLogs("mashin", level=client_module.TRACE) as captured:
            await client.state()
            await client.state()
        self.assertIn("TRACE:mashin:Playback position:", "\n".join(captured.output))

    async def test_pause_accounts_for_drained_frames_with_lagging_device_clock(self):
        output = FakeOutput()
        output.load(descriptor(), True)
        stream = output.current
        output.accept(stream, flac_chunk(stream))
        output.prepare()
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
        client = client_module.Client(client_module.arguments(["localhost:8095"]), output)
        client.send = AsyncMock()
        with self.assertLogs("mashin", level="ERROR"):
            await client.command(
                "set_media", {"command_id": 1, "stream": descriptor(), "autoplay": True}
            )
        self.assertEqual(client.send.await_args_list[0].args[1]["status"], "rejected")
        self.assertIsNone(output.current)
        self.assertEqual(output.snapshot()["output_state"], "idle")

    async def test_activation_reports_receipt_before_b_state_and_resumes_it(self):
        output = FakeOutput()
        output.load(descriptor(), True)
        current = output.current
        output.next = successor = stream_from(descriptor())
        output.next_revision = 1
        output.accept(successor, flac_chunk(successor))
        client = client_module.Client(client_module.arguments(["localhost:8095"]), output)
        client.send = AsyncMock()
        await client.command(
            "activate_prepared",
            {
                "command_id": 1,
                "current_stream_id": current.identifier,
                "stream_id": successor.identifier,
                "next_revision": 1,
            },
        )
        self.assertEqual(
            [call.args[0] for call in client.send.await_args_list],
            [
                "command_ack",
                "prepared_activated",
                "player_state",
            ],
        )
        receipt = client.send.await_args_list[1].args[1]["activation"]
        self.assertEqual(receipt["previous_stream_id"], current.identifier)
        self.assertEqual(receipt["stream_id"], successor.identifier)
        self.assertEqual(client.resume()["activation"], receipt)
        self.assertIs(output.current, successor)
        await client.command(
            "activate_prepared",
            {
                "command_id": 2,
                "current_stream_id": current.identifier,
                "stream_id": successor.identifier,
                "next_revision": 1,
            },
        )
        self.assertIs(output.current, successor)
        self.assertEqual(client.resume()["activation"], receipt)
        output.stop()

    async def test_commands_and_duplicate_load(self):
        output = FakeOutput()
        client = client_module.Client(client_module.arguments(["localhost:8095"]), output)
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
            output.accept(stream, flac_chunk(stream))
        output.prepare()
        output._callback(
            np.empty((48000, 2), np.float32),
            48000,
            SimpleNamespace(outputBufferDacTime=1.0),
            None,
        )
        output.output.time = 2.0
        self.assertEqual([kind for kind, _ in output.poll()], ["track_ended", "track_started"])
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
            "protocol_version": 2,
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
        if getattr(self, "reject_resume", False):
            error = self.envelope(
                "error",
                {
                    "error": client_module.protocol_error(
                        "stale_playback", "Playback is no longer current"
                    )
                },
            )
            error["connection_epoch"] = None
            await connection.send_json(error)
            await connection.close()
            return connection
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
        self.output.prepare()
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

    async def test_rejected_resume_discards_cache_and_allows_idle_reconnect(self):
        await self.load(autoplay=False)
        await self.client.control.close()
        with self.assertRaises(ConnectionError):
            await asyncio.wait_for(self.task, 5)
        self.reject_resume = True
        self.task = asyncio.create_task(self.client.connect(self.session))
        with (
            self.assertLogs("mashin", level="WARNING"),
            self.assertRaisesRegex(ConnectionError, "Server cannot resume retained playback"),
        ):
            await asyncio.wait_for(self.task, 5)
        self.assertIsNone(self.output.current)
        self.assertEqual(self.output.cursors(), [])
        self.reject_resume = False
        self.audio_open.clear()
        self.epoch = str(uuid4())
        self.task = asyncio.create_task(self.client.connect(self.session))
        await asyncio.wait_for(self.audio_open.wait(), 5)
        resume = self.hellos[-1]["payload"]["resume_state"]
        self.assertIsNone(resume["current_stream_id"])
        self.assertEqual(resume["output_state"], "idle")
        self.assertEqual(resume["received_streams"], [])

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
        self.assertEqual(self.audio_hello["payload"]["received_streams"], self.output.cursors())
        self.assertEqual(self.output.snapshot()["output_state"], "paused")


if __name__ == "__main__":
    unittest.main()
