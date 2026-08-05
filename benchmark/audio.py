from __future__ import annotations

from dataclasses import dataclass
import io
import subprocess
import tempfile
import wave

import numpy as np

from .models import AudioFormat


@dataclass(frozen=True)
class OnsetResult:
    onset_sample: int
    onset_ms: float
    threshold: float
    noise_floor: float
    algorithm: str
    low_threshold: float | None = None
    speech_level: float | None = None


def decode_audio(payload: bytes, fmt: AudioFormat) -> np.ndarray:
    encoding = fmt.encoding.lower()
    if encoding in {"pcm", "pcm_s16le", "pcm16", "raw"}:
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        if fmt.channels > 1:
            samples = samples[: len(samples) - len(samples) % fmt.channels]
            samples = samples.reshape(-1, fmt.channels).mean(axis=1)
        return samples
    if encoding in {"pcm_f32le", "f32le"}:
        samples = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=False)
        if fmt.channels > 1:
            samples = samples[: len(samples) - len(samples) % fmt.channels]
            samples = samples.reshape(-1, fmt.channels).mean(axis=1)
        return samples
    if encoding in {"wav", "wave"}:
        with wave.open(io.BytesIO(payload), "rb") as wav:
            width = wav.getsampwidth()
            if width != 2:
                raise ValueError(f"only 16-bit WAV is supported directly, got width={width}")
            channels = wav.getnchannels()
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            samples = samples.astype(np.float32) / 32768.0
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)
            return samples
    return _decode_with_ffmpeg(payload, encoding, fmt.sample_rate_hz)


def _decode_with_ffmpeg(payload: bytes, encoding: str, sample_rate_hz: int) -> np.ndarray:
    suffix = "." + ("mp3" if "mp3" in encoding else encoding.replace("_", "."))
    with tempfile.NamedTemporaryFile(suffix=suffix) as source:
        source.write(payload)
        source.flush()
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            source.name,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate_hz),
            "pipe:1",
        ]
        result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"ffmpeg could not decode {encoding}: {detail}")
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0


def detect_voice_onset_v1(
    samples: np.ndarray,
    sample_rate_hz: int,
    *,
    frame_ms: float = 1.0,
    noise_window_ms: float = 12.0,
    density_window_ms: float = 10.0,
    threshold_factor: float = 3.0,
    threshold_floor: float = 0.002,
    active_ratio: float = 0.60,
    min_active_ms: float = 4.0,
) -> OnsetResult:
    if samples.size == 0:
        raise ValueError("audio is empty")
    samples = samples.astype(np.float32, copy=False) - float(np.mean(samples))
    frame = max(1, round(sample_rate_hz * frame_ms / 1000))
    usable = (samples.size // frame) * frame
    if usable == 0:
        padded = np.pad(samples, (0, frame - samples.size))
        envelopes = np.array([np.mean(np.abs(padded))])
    else:
        envelopes = np.mean(np.abs(samples[:usable].reshape(-1, frame)), axis=1)
    noise_frames = max(1, round(noise_window_ms / frame_ms))
    noise_floor = float(np.median(envelopes[:noise_frames]))
    threshold = max(threshold_floor, noise_floor * threshold_factor)
    active = envelopes >= threshold
    density_frames = max(1, round(density_window_ms / frame_ms))
    min_frames = max(1, round(min_active_ms / frame_ms))
    counts = np.convolve(active.astype(np.int16), np.ones(density_frames, dtype=np.int16), "same")
    dense = counts / density_frames >= active_ratio
    for dense_index in np.flatnonzero(dense):
        left = max(0, dense_index - density_frames // 2)
        right = min(active.size, dense_index + (density_frames + 1) // 2)
        run = 0
        for index in range(left, right):
            run = run + 1 if active[index] else 0
            if run >= min_frames:
                onset_frame = index - run + 1
                onset_sample = onset_frame * frame
                return OnsetResult(
                    onset_sample=onset_sample,
                    onset_ms=onset_sample * 1000.0 / sample_rate_hz,
                    threshold=threshold,
                    noise_floor=noise_floor,
                    algorithm="dual_abs_density_v1",
                )
    raise ValueError("no audible voice onset detected")


def detect_voice_onset(
    samples: np.ndarray,
    sample_rate_hz: int,
    *,
    frame_ms: float = 2.0,
    high_density_window_ms: float = 10.0,
    high_active_ratio: float = 0.60,
    low_density_window_ms: float = 20.0,
    low_active_ratio: float = 0.80,
    min_low_active_ms: float = 10.0,
    high_absolute_floor: float = 0.0005,
    low_absolute_floor: float = 0.0001,
    high_speech_ratio: float = 0.030,
    low_speech_ratio: float = 0.005,
    max_backtrack_gap_ms: float = 12.0,
) -> OnsetResult:
    """Detect audible TTS onset with relative dual thresholds and hysteresis.

    Unlike v1, this detector never assumes the first frames are background noise.
    A sustained low-energy consonant can establish onset directly, while a strong
    speech anchor is allowed to backtrack across short sub-threshold gaps. The
    sustained-density requirement rejects isolated clicks.
    """
    if samples.size == 0:
        raise ValueError("audio is empty")
    samples = samples.astype(np.float32, copy=False) - float(np.median(samples))
    frame = max(1, round(sample_rate_hz * frame_ms / 1000))
    envelopes = _rms_envelopes(samples, frame)
    speech_level = float(np.percentile(envelopes, 95))
    high_threshold = max(high_absolute_floor, speech_level * high_speech_ratio)
    low_threshold = max(low_absolute_floor, speech_level * low_speech_ratio)

    high_onset = _first_dense_onset(
        envelopes >= high_threshold,
        frame_ms=frame_ms,
        density_window_ms=high_density_window_ms,
        active_ratio=high_active_ratio,
        min_active_ms=min(frame_ms * 2, high_density_window_ms),
    )
    low_onset = _first_dense_onset(
        envelopes >= low_threshold,
        frame_ms=frame_ms,
        density_window_ms=low_density_window_ms,
        active_ratio=low_active_ratio,
        min_active_ms=min_low_active_ms,
    )
    if high_onset is None and low_onset is None:
        raise ValueError("no audible voice onset detected")

    candidates = [index for index in (low_onset, high_onset) if index is not None]
    if high_onset is not None:
        candidates.append(
            _backtrack_onset(
                envelopes,
                high_onset,
                low_threshold,
                frame_ms=frame_ms,
                max_gap_ms=max_backtrack_gap_ms,
            )
        )
    onset_frame = min(candidates)
    onset_sample = onset_frame * frame
    noise_frames = max(1, round(12.0 / frame_ms))
    noise_floor = float(np.median(envelopes[:noise_frames]))
    return OnsetResult(
        onset_sample=onset_sample,
        onset_ms=onset_sample * 1000.0 / sample_rate_hz,
        threshold=high_threshold,
        noise_floor=noise_floor,
        algorithm="dual_energy_hysteresis_v2",
        low_threshold=low_threshold,
        speech_level=speech_level,
    )


def _rms_envelopes(samples: np.ndarray, frame: int) -> np.ndarray:
    usable = (samples.size // frame) * frame
    if usable == 0:
        padded = np.pad(samples, (0, frame - samples.size))
        return np.array([np.sqrt(np.mean(np.square(padded)))])
    framed = samples[:usable].reshape(-1, frame)
    return np.sqrt(np.mean(np.square(framed), axis=1))


def _first_dense_onset(
    active: np.ndarray,
    *,
    frame_ms: float,
    density_window_ms: float,
    active_ratio: float,
    min_active_ms: float,
) -> int | None:
    density_frames = max(1, round(density_window_ms / frame_ms))
    min_frames = max(1, round(min_active_ms / frame_ms))
    counts = np.convolve(active.astype(np.int16), np.ones(density_frames, dtype=np.int16), "same")
    dense = counts / density_frames >= active_ratio
    for dense_index in np.flatnonzero(dense):
        left = max(0, dense_index - density_frames // 2)
        right = min(active.size, dense_index + (density_frames + 1) // 2)
        run = 0
        for index in range(left, right):
            run = run + 1 if active[index] else 0
            if run >= min_frames:
                return index - run + 1
    return None


def _backtrack_onset(
    envelopes: np.ndarray,
    anchor: int,
    low_threshold: float,
    *,
    frame_ms: float,
    max_gap_ms: float,
) -> int:
    start = anchor
    gap = 0
    max_gap_frames = max(0, round(max_gap_ms / frame_ms))
    for index in range(anchor, -1, -1):
        if envelopes[index] >= low_threshold:
            start = index
            gap = 0
            continue
        gap += 1
        if gap > max_gap_frames:
            break
    return start


def audio_duration_ms(samples: np.ndarray, sample_rate_hz: int) -> float:
    return samples.size * 1000.0 / sample_rate_hz
