"""Strict RIFF/WAVE parser for integer PCM audio.

The parser reads the container structure exactly and raises :class:`WavError`
for anything malformed instead of guessing:

- the file starts with ``RIFF`` and ``WAVE``, and the RIFF size equals the
  file size minus 8;
- chunks tile the RIFF body exactly (each chunk padded to an even size), with
  no truncated chunk and no trailing bytes;
- exactly one ``fmt `` chunk, which precedes exactly one ``data`` chunk;
- the ``fmt `` fields are internally consistent (``block_align`` and
  ``byte_rate`` match channels, bits, and sample rate);
- the ``data`` size is a whole number of frames.

Only format tag 1 (integer PCM) samples are decoded. Whether the sample rate,
channel count, or bit depth is acceptable is a policy decision left to the
caller. Other chunks (for example ``LIST``) are skipped and reported.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "WAVE_FORMAT_IEEE_FLOAT",
    "WAVE_FORMAT_PCM",
    "WavData",
    "WavError",
    "encode_wav_float32",
    "parse_wav",
]

WAVE_FORMAT_PCM = 1
WAVE_FORMAT_IEEE_FLOAT = 3


class WavError(ValueError):
    """Raised when bytes are not a well-formed WAVE file."""


@dataclass(frozen=True)
class WavData:
    """Header fields and raw sample bytes of a parsed WAVE file."""

    format_tag: int
    num_channels: int
    sample_rate_hz: int
    bits_per_sample: int
    block_align: int
    byte_rate: int
    sample_bytes: bytes
    extra_chunks: tuple[str, ...]

    @property
    def num_frames(self) -> int:
        return len(self.sample_bytes) // self.block_align

    @property
    def sample_format(self) -> str:
        if self.format_tag == WAVE_FORMAT_PCM:
            return f"PCM_{self.bits_per_sample}"
        return f"FORMAT_TAG_{self.format_tag}_{self.bits_per_sample}BIT"

    def samples_int16(self) -> npt.NDArray[np.int16]:
        """Decode samples as little-endian signed 16-bit integers (PCM_16 only)."""
        if self.format_tag != WAVE_FORMAT_PCM or self.bits_per_sample != 16:
            raise WavError(f"cannot decode {self.sample_format} as PCM_16")
        return np.frombuffer(self.sample_bytes, dtype="<i2")

    def samples_float32(self) -> npt.NDArray[np.float32]:
        """Decode samples as little-endian IEEE float32 (format tag 3, 32 bits only)."""
        if self.format_tag != WAVE_FORMAT_IEEE_FLOAT or self.bits_per_sample != 32:
            raise WavError(f"cannot decode {self.sample_format} as float32")
        return np.frombuffer(self.sample_bytes, dtype="<f4")


def encode_wav_float32(samples: npt.NDArray[np.float32], sample_rate_hz: int) -> bytes:
    """Mono IEEE-float32 WAVE bytes (``fmt `` chunk of 16 bytes, then ``data``)."""
    if samples.ndim != 1:
        raise WavError("only mono samples can be encoded")
    data = samples.astype("<f4").tobytes()
    rate = sample_rate_hz
    fmt = struct.pack("<HHIIHH", WAVE_FORMAT_IEEE_FLOAT, 1, rate, rate * 4, 4, 32)
    body = b"WAVE" + struct.pack("<4sI", b"fmt ", 16) + fmt
    body += struct.pack("<4sI", b"data", len(data)) + data  # float32 data is always even-sized
    return struct.pack("<4sI", b"RIFF", len(body)) + body


def parse_wav(data: bytes) -> WavData:
    """Parse ``data`` as a WAVE file, raising :class:`WavError` if it is malformed."""
    if len(data) < 12:
        raise WavError(f"file too short for a RIFF header ({len(data)} bytes)")
    riff, riff_size, wave = struct.unpack("<4sI4s", data[:12])
    if riff != b"RIFF" or wave != b"WAVE":
        raise WavError("missing RIFF/WAVE signature")
    if riff_size != len(data) - 8:
        raise WavError(f"RIFF size {riff_size} does not match file size {len(data)} - 8")

    fmt: tuple[int, int, int, int, int, int] | None = None
    sample_bytes: bytes | None = None
    extra: list[str] = []
    pos = 12
    while pos < len(data):
        if pos + 8 > len(data):
            raise WavError(f"truncated chunk header at byte {pos}")
        chunk_id, size = struct.unpack("<4sI", data[pos : pos + 8])
        body_start = pos + 8
        body_end = body_start + size
        padded_end = body_end + (size & 1)
        if padded_end > len(data):
            raise WavError(f"chunk {chunk_id!r} at byte {pos} extends past end of file")
        body = data[body_start:body_end]
        if chunk_id == b"fmt ":
            if fmt is not None:
                raise WavError("more than one fmt chunk")
            if sample_bytes is not None:
                raise WavError("fmt chunk after data chunk")
            if size < 16:
                raise WavError(f"fmt chunk too short ({size} bytes)")
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif chunk_id == b"data":
            if sample_bytes is not None:
                raise WavError("more than one data chunk")
            if fmt is None:
                raise WavError("data chunk before fmt chunk")
            sample_bytes = body
        else:
            extra.append(chunk_id.decode("latin-1"))
        pos = padded_end

    if fmt is None:
        raise WavError("missing fmt chunk")
    if sample_bytes is None:
        raise WavError("missing data chunk")
    format_tag, channels, rate, byte_rate, block_align, bits = fmt
    if channels < 1 or rate < 1 or bits < 1:
        raise WavError(f"invalid fmt fields: channels={channels} rate={rate} bits={bits}")
    expected_align = channels * ((bits + 7) // 8)
    if block_align != expected_align:
        raise WavError(f"block_align {block_align} != channels * bytes_per_sample {expected_align}")
    if byte_rate != rate * block_align:
        raise WavError(f"byte_rate {byte_rate} != sample_rate * block_align {rate * block_align}")
    if len(sample_bytes) % block_align:
        raise WavError(f"data size {len(sample_bytes)} is not a multiple of {block_align}")
    return WavData(
        format_tag=format_tag,
        num_channels=channels,
        sample_rate_hz=rate,
        bits_per_sample=bits,
        block_align=block_align,
        byte_rate=byte_rate,
        sample_bytes=sample_bytes,
        extra_chunks=tuple(extra),
    )
