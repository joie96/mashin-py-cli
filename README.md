# mashin-py-cli

Minimal Python console receiver for the Music Assistant **Mashin** player provider.
Audio plays on the system's default output device. Select media and control the
registered player in Music Assistant; there are no local playback controls.

## Install and Run

Requires Python 3.12+ and a Music Assistant server with the Mashin provider enabled.
Run these commands in this repository:

```console
python -m pip install -r requirements.txt
python mashin-py-cli.py http://192.168.1.10:8095 --name "Desktop"
```

Alternatively, `uv run` installs the dependencies from the script's inline metadata:

```console
uv run mashin-py-cli.py http://192.168.1.10:8095 --name "Desktop"
```

The client registers automatically. Choose its player in MA and start playback there.
Stop the receiver with Ctrl+C. Audio is decoded in memory, never saved to disk.

## Options

```console
python mashin-py-cli.py --list-devices
python mashin-py-cli.py 192.168.1.10:8095 --output-device 3
python mashin-py-cli.py 192.168.1.10:8095 --device-id desktop --profile opus_192k
python mashin-py-cli.py 192.168.1.10:8095 --codec opus --buffer-capacity 8 --debug
python mashin-py-cli.py --help
```

- `server`: MA base URL, accepting HTTP/HTTPS or WS/WSS. HTTP is converted to WS;
	HTTPS to WSS with certificate verification. A reverse-proxy base path is preserved.
- `--name`: player name shown in MA.
- `--device-id`: stable ASCII identity, defaulting to an ID derived from the hostname.
	Keep it unchanged across reconnects; use separate IDs for simultaneous clients.
- `--codec` (alias `--profile`): `flac` (default), `opus_96k`, `opus_128k`,
  `opus_192k`, or `opus_256k`. `opus` is shorthand for `opus_128k`.
- `--buffer-capacity`: budget for original encoded server chunks in whole MiB,
  minimum 1, default 4. Current and successor tracks share this budget, including
  the chunk headers. Receipt ACKs confirm retained encoded bytes, not decoding.
  Byte credits reflect available space, capped at the protocol's payload limit.
  This is not a duration or prebuffer delay; server limits can constrain occupancy.
  A decoder worker prepares at most two current-track chunks as a small PCM
  working set for the sound device. Queued successor chunks remain encoded.
- `--output-device`: device index or name substring; omitted uses the default output.
- `--debug`: debug logging with handshake diagnostics and a buffer summary every
    five seconds: occupied KiB, cached chunk count and remaining audio seconds.
    Totals include current and successor tracks; duration excludes frames already
    submitted to the audio device.
- `--trace`: detailed chunk, decoding, buffer-credit, protocol and position logs;
    includes all debug output. Without either flag, logging uses INFO for connection,
    command, playback-state and error messages. `--verbose` is not supported.

MA commands for pause, resume, stop, volume and mute are applied to local playback.
The receiver reports DAC-based position and track completion, bounded buffer
capacity and encoded-chunk receipts. It reconnects automatically, retaining buffered audio
for up to 30 seconds after detecting a disconnect. Successor tracks are supported,
but device reopening between tracks is not guaranteed gapless.

The Mashin endpoints are token-free and intended for a trusted LAN. Do not expose
them directly to the Internet. WSS requires TLS on MA's existing webserver or proxy.
On Linux, install PortAudio if unavailable (for example `libportaudio2` on Debian).
The client uses PyAV's codec libraries; a separate FFmpeg executable is not required
when using the supplied binary wheels.

## Tests

```console
python -m unittest discover -p test_client.py -v
```

Tests use synthetic audio, a fake output clock and a local WebSocket server.
They do not require speakers or a running MA instance.
