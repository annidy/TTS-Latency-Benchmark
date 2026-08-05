from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import random
import time
import uuid
from typing import Any

from .audio import (
    OnsetResult,
    audio_duration_ms,
    decode_audio,
    detect_voice_onset,
    detect_voice_onset_v1,
)
from .clock import Clock, MonotonicClock
from .models import EventType, ProviderEvent, SampleResult, SynthesisRequest
from .adapters import ModelAdapter, registry


class BenchmarkEngine:
    def __init__(self, clock: Clock | None = None, *, retain_audio: bool = False) -> None:
        self.clock = clock or MonotonicClock()
        self.retain_audio = retain_audio

    async def run_sample(
        self,
        *,
        run_id: str,
        client: dict[str, Any],
        provider: dict[str, Any],
        profile: dict[str, Any],
        text: dict[str, Any],
        adapter: ModelAdapter | None = None,
    ) -> SampleResult:
        sample_id = str(uuid.uuid4())
        request = SynthesisRequest(
            request_id=sample_id,
            text=str(text["text"]),
            language=text.get("language"),
            voice=provider.get("voice", {}).get("value"),
        )
        owns_adapter = adapter is None
        adapter = adapter or registry.create(
            str(provider["kind"]), str(provider["id"]), provider, self.clock
        )
        started_ns = self.clock.now_ns()
        started_at = datetime.now(timezone.utc).isoformat()
        measurement_protocol = adapter.measurement_protocol(profile)
        events = [ProviderEvent(EventType.REQUEST_STARTED, started_ns)]
        chunks: list[ProviderEvent] = []
        result = SampleResult(
            run_id=run_id,
            sample_id=sample_id,
            client_id=str(client["id"]),
            client_region=str(client.get("region", "unknown")),
            provider_id=str(provider["id"]),
            provider_kind=str(provider["kind"]),
            profile_id=str(profile["id"]),
            text_id=str(text["id"]),
            text=str(text["text"]),
            connection_mode=str(profile.get("connection_mode", "new")),
            transport=str(provider.get("transport", "unknown")),
            protocol=str(measurement_protocol["name"]),
            status="failed",
            started_at=started_at,
            metadata={
                "text_language": text.get("language"),
                "text_profile": text.get("profile"),
                "onset_algorithm": "dual_energy_hysteresis_v2",
                "onset_algorithms_retained": [
                    "dual_abs_density_v1",
                    "dual_energy_hysteresis_v2",
                ],
                "measurement_protocol": measurement_protocol,
                "client": {
                    key: value for key, value in client.items() if key not in {"runner", "enabled"}
                },
                "provider_settings": {
                    "endpoint": {
                        key: provider.get("endpoint", {}).get(key)
                        for key in (
                            "url",
                            "service_region",
                            "region_policy",
                            "region_scope",
                            "region_source",
                        )
                        if provider.get("endpoint", {}).get(key) is not None
                    },
                    "model": provider.get("model"),
                    "voice": provider.get("voice"),
                    "audio": provider.get("audio"),
                    "request": {
                        key: value
                        for key, value in provider.get("request", {}).items()
                        if key not in {"overrides"}
                    },
                },
            },
        )
        try:
            async for event in adapter.synthesize(request):
                if event.at_ns < started_ns:
                    raise ValueError("provider emitted an event before request_started")
                events.append(event)
                if event.type == EventType.AUDIO_CHUNK and event.audio:
                    chunks.append(event)
            self._finalize_result(result, events, chunks, started_ns)
        except Exception as error:  # provider failures are benchmark data
            result.error_code = _error_code(error)
            result.error = str(error)
            result.total_latency_ms = (self.clock.now_ns() - started_ns) / 1_000_000
            events.append(
                ProviderEvent(
                    EventType.REQUEST_FAILED,
                    self.clock.now_ns(),
                    data={"code": result.error_code, "message": result.error},
                )
            )
        finally:
            if owns_adapter:
                await adapter.close()
            result.events = [event.public_dict(started_ns) for event in events]
        return result

    def _finalize_result(
        self,
        result: SampleResult,
        events: list[ProviderEvent],
        chunks: list[ProviderEvent],
        started_ns: int,
    ) -> None:
        if not chunks:
            raise ValueError("provider stream completed without audio")
        first_chunk = chunks[0]
        finished = _first(events, EventType.STREAM_FINISHED) or events[-1]
        fmt = first_chunk.audio_format
        if fmt is None:
            raise ValueError("audio chunk is missing AudioFormat")
        if any(chunk.audio_format != fmt for chunk in chunks):
            raise ValueError("audio format changed within one provider stream")
        payload = b"".join(chunk.audio or b"" for chunk in chunks)
        decoded = decode_audio(payload, fmt)
        onset_v1: OnsetResult | None = None
        onset_v1_error: str | None = None
        try:
            onset_v1 = detect_voice_onset_v1(decoded, fmt.sample_rate_hz)
        except ValueError as error:
            # The retained legacy detector must not fail a sample accepted by the
            # canonical v2 detector.
            onset_v1_error = str(error)
        onset_v2 = detect_voice_onset(decoded, fmt.sample_rate_hz)
        # TTFB is deliberately anchored to the first non-empty audio chunk. Provider
        # metadata/control messages are retained as events but never count as TTFB.
        ttfb = (first_chunk.at_ns - started_ns) / 1_000_000
        metrics_v1 = (
            _onset_metrics(onset_v1, chunks, fmt, started_ns, ttfb)
            if onset_v1 is not None
            else None
        )
        metrics_v2 = _onset_metrics(onset_v2, chunks, fmt, started_ns, ttfb)
        audio_clock_ttfa = metrics_v2["ttfa_audio_clock_ms"]
        available_ttfa = metrics_v2["ttfa_available_ms"]
        canonical_ttfa = max(audio_clock_ttfa, available_ttfa or audio_clock_ttfa)
        duration = audio_duration_ms(decoded, fmt.sample_rate_hz)
        total = (finished.at_ns - started_ns) / 1_000_000
        server_timing = _first(events, EventType.SERVER_TIMING)
        result.status = "completed"
        result.ttfb_ms = ttfb
        result.initial_silence_ms = onset_v2.onset_ms
        result.initial_silence_v1_ms = onset_v1.onset_ms if onset_v1 else None
        result.initial_silence_v2_ms = onset_v2.onset_ms
        result.ttfa_audio_clock_ms = audio_clock_ttfa
        result.ttfa_available_ms = available_ttfa
        result.ttfa_ms = canonical_ttfa
        result.ttfa_v1_ms = metrics_v1["ttfa_ms"] if metrics_v1 else None
        result.ttfa_v2_ms = metrics_v2["ttfa_ms"]
        result.total_latency_ms = total
        result.audio_duration_ms = duration
        result.rtf = total / duration if duration > 0 else None
        result.server_ttfa_ms = (
            float(server_timing.data["ttfa_ms"])
            if server_timing and server_timing.data.get("ttfa_ms") is not None
            else None
        )
        result.audio_format = asdict(fmt)
        if self.retain_audio:
            result.audio_payload = payload
        result.metadata.update(
            {
                "onset_sample": onset_v2.onset_sample,
                "onset_threshold": onset_v2.threshold,
                "onset_low_threshold": onset_v2.low_threshold,
                "onset_noise_floor": onset_v2.noise_floor,
                "onset_speech_level": onset_v2.speech_level,
                "onset_detectors": {
                    "dual_abs_density_v1": (
                        _onset_metadata(onset_v1, metrics_v1)
                        if onset_v1 and metrics_v1
                        else {"error": onset_v1_error}
                    ),
                    onset_v2.algorithm: _onset_metadata(onset_v2, metrics_v2),
                },
                "audio_bytes": len(payload),
                "chunk_count": len(chunks),
            }
        )


async def run_local_matrix(
    *,
    run_id: str,
    client: dict[str, Any],
    providers: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    texts: list[dict[str, Any]],
    seed: int,
    clock: Clock | None = None,
    retain_audio: bool = False,
) -> list[SampleResult]:
    engine = BenchmarkEngine(clock, retain_audio=retain_audio)
    rng = random.Random(seed)
    output: list[SampleResult] = []
    for profile in profiles:
        warmups = int(profile.get("warmup_runs", 0))
        repetitions = int(profile.get("repetitions", 1))
        persistent = profile.get("connection_mode") == "persistent"
        adapters: dict[str, ModelAdapter] = {}
        if persistent:
            adapters = {
                provider["id"]: registry.create(
                    provider["kind"], provider["id"], provider, engine.clock
                )
                for provider in providers
            }

        async def reset_persistent_adapter(provider: dict[str, Any]) -> None:
            """Discard a broken persistent transport before it is used again."""
            if not persistent:
                return
            provider_id = provider["id"]
            previous = adapters.get(provider_id)
            if previous is not None:
                try:
                    await previous.close()
                except Exception:
                    # Reconnection must not be prevented by a half-closed socket.
                    pass
            adapters[provider_id] = registry.create(
                provider["kind"], provider_id, provider, engine.clock
            )

        async def warm_provider(provider: dict[str, Any]) -> None:
            label = _provider_label(provider)
            warmup_retries = int(profile.get("warmup_retries", 0))
            retry_backoff_ms = float(profile.get("retry_backoff_seconds", 0)) * 1000
            if warmup_retries < 0 or retry_backoff_ms < 0:
                raise ValueError("warmup retries and retry backoff cannot be negative")
            for warmup_index in range(warmups):
                for retry_index in range(warmup_retries + 1):
                    _emit_progress(
                        model=label,
                        phase="warmup",
                        request=f"{warmup_index + 1}/{warmups}",
                        status="started",
                        attempt=str(retry_index + 1),
                    )
                    warmup_request = SynthesisRequest(
                        request_id=f"warmup-{warmup_index}-try-{retry_index}",
                        text="Hello.",
                    )
                    target = adapters.get(provider["id"]) or registry.create(
                        provider["kind"], provider["id"], provider, engine.clock
                    )
                    try:
                        await target.warmup(warmup_request)
                        _emit_progress(
                            model=label,
                            phase="warmup",
                            request=f"{warmup_index + 1}/{warmups}",
                            status="completed",
                            attempt=str(retry_index + 1),
                        )
                        break
                    except Exception:
                        exhausted = retry_index == warmup_retries
                        _emit_progress(
                            model=label,
                            phase="warmup",
                            request=f"{warmup_index + 1}/{warmups}",
                            status="failed" if exhausted else "retrying",
                            attempt=str(retry_index + 1),
                        )
                        if exhausted:
                            raise
                        await reset_persistent_adapter(provider)
                        await engine.clock.sleep_ms(retry_backoff_ms)
                    finally:
                        if not persistent:
                            await target.close()

        try:
            iteration_order = str(profile.get("iteration_order", "interleaved"))
            provider_execution = str(profile.get("provider_execution", "serial"))
            if iteration_order == "provider_major":
                if provider_execution != "serial":
                    raise ValueError("provider_major iteration requires serial execution")
                provider_order = str(profile.get("provider_order", "configured"))
                if provider_order not in {"configured", "shuffled"}:
                    raise ValueError("provider_order must be configured or shuffled")
                ordered_providers = list(providers)
                if provider_order == "shuffled":
                    rng.shuffle(ordered_providers)
                measurement_retries = int(profile.get("measurement_retries", 0))
                retry_backoff_ms = float(profile.get("retry_backoff_seconds", 0)) * 1000
                if measurement_retries < 0:
                    raise ValueError("measurement_retries cannot be negative")
                for provider in ordered_providers:
                    # Warm persistent transports immediately before this model's
                    # measurements. Warming every model up front lets later WebSocket
                    # connections expire while earlier models run long serial batches.
                    await warm_provider(provider)
                    label = _provider_label(provider)
                    total_requests = repetitions * len(texts)
                    attempted_requests = 0
                    successful_requests = 0
                    model_started = time.monotonic()
                    _emit_progress(
                        model=label,
                        phase="measure",
                        request=f"0/{total_requests}",
                        status="started",
                    )
                    for repetition in range(repetitions):
                        for text in texts:
                            failed_attempts: list[SampleResult] = []
                            for retry_index in range(measurement_retries + 1):
                                sample = await engine.run_sample(
                                    run_id=run_id,
                                    client=client,
                                    provider=provider,
                                    profile=profile,
                                    text=text,
                                    adapter=adapters.get(provider["id"]),
                                )
                                sample.metadata["repetition"] = repetition
                                sample.metadata["retry_index"] = retry_index
                                output.append(sample)
                                attempted_requests += 1
                                if sample.status == "completed":
                                    successful_requests += 1
                                    for failed in failed_attempts:
                                        failed.metadata["retry_recovered"] = True
                                    _emit_sample_progress(
                                        model=label,
                                        completed=successful_requests,
                                        total=total_requests,
                                        successes=successful_requests,
                                        attempts=attempted_requests,
                                        sample=sample,
                                    )
                                    break
                                failed_attempts.append(sample)
                                _emit_sample_progress(
                                    model=label,
                                    completed=successful_requests,
                                    total=total_requests,
                                    successes=successful_requests,
                                    attempts=attempted_requests,
                                    sample=sample,
                                )
                                if retry_index < measurement_retries:
                                    await engine.clock.sleep_ms(retry_backoff_ms)
                                    # A failed turn can leave a persistent WebSocket in
                                    # an unusable protocol state. Recreate it and require
                                    # the complete excluded warmup sequence again before
                                    # accepting another measured sample.
                                    await reset_persistent_adapter(provider)
                                    await warm_provider(provider)
                    _emit_progress(
                        model=label,
                        phase="model",
                        request=f"{successful_requests}/{total_requests}",
                        status=("completed" if successful_requests == total_requests else "failed"),
                        successes=f"{successful_requests}/{total_requests}",
                        attempts=str(attempted_requests),
                        elapsed_s=f"{time.monotonic() - model_started:.1f}",
                    )
            elif iteration_order == "interleaved":
                for provider in providers:
                    await warm_provider(provider)
                total_requests = repetitions * len(texts)
                completed_by_provider = {provider["id"]: 0 for provider in providers}
                successes_by_provider = {provider["id"]: 0 for provider in providers}
                for repetition in range(repetitions):
                    for text in texts:
                        ordered = list(providers)
                        rng.shuffle(ordered)
                        if provider_execution == "parallel":
                            batch = await asyncio.gather(
                                *[
                                    engine.run_sample(
                                        run_id=run_id,
                                        client=client,
                                        provider=provider,
                                        profile=profile,
                                        text=text,
                                        adapter=adapters.get(provider["id"]),
                                    )
                                    for provider in ordered
                                ]
                            )
                            output.extend(batch)
                            for provider, sample in zip(ordered, batch, strict=True):
                                sample.metadata["repetition"] = repetition
                                provider_id = provider["id"]
                                completed_by_provider[provider_id] += 1
                                successes_by_provider[provider_id] += sample.status == "completed"
                                _emit_sample_progress(
                                    model=_provider_label(provider),
                                    completed=completed_by_provider[provider_id],
                                    total=total_requests,
                                    successes=successes_by_provider[provider_id],
                                    sample=sample,
                                )
                        else:
                            for provider in ordered:
                                sample = await engine.run_sample(
                                    run_id=run_id,
                                    client=client,
                                    provider=provider,
                                    profile=profile,
                                    text=text,
                                    adapter=adapters.get(provider["id"]),
                                )
                                sample.metadata["repetition"] = repetition
                                output.append(sample)
                                provider_id = provider["id"]
                                completed_by_provider[provider_id] += 1
                                successes_by_provider[provider_id] += sample.status == "completed"
                                _emit_sample_progress(
                                    model=_provider_label(provider),
                                    completed=completed_by_provider[provider_id],
                                    total=total_requests,
                                    successes=successes_by_provider[provider_id],
                                    sample=sample,
                                )
            else:
                raise ValueError(f"unsupported iteration_order: {iteration_order}")
        finally:
            await asyncio.gather(*(adapter.close() for adapter in adapters.values()))
    return output


def _provider_label(provider: dict[str, Any]) -> str:
    return str(provider.get("kind") or provider.get("id") or "unknown")


def _emit_sample_progress(
    *,
    model: str,
    completed: int,
    total: int,
    successes: int,
    attempts: int | None = None,
    sample: SampleResult,
) -> None:
    attempt_count = completed if attempts is None else attempts
    fields: dict[str, str] = {
        "model": model,
        "phase": "measure",
        "request": f"{completed}/{total}",
        "status": sample.status,
        "successes": f"{successes}/{attempt_count}",
        "attempts": str(attempt_count),
    }
    if sample.ttfa_ms is not None:
        fields["ttfa_ms"] = f"{sample.ttfa_ms:.1f}"
    if sample.total_latency_ms is not None:
        fields["total_ms"] = f"{sample.total_latency_ms:.1f}"
    if sample.error_code:
        fields["error_code"] = sample.error_code
    _emit_progress(**fields)


def _emit_progress(**fields: str) -> None:
    rendered = " ".join(f"{key}={json_quote(value)}" for key, value in fields.items())
    print(f"[progress] {rendered}", flush=True)


def json_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _first(events: list[ProviderEvent], event_type: EventType) -> ProviderEvent | None:
    return next((event for event in events if event.type == event_type), None)


def _onset_available_ms(
    chunks: list[ProviderEvent], fmt: Any, onset_sample: int, started_ns: int
) -> float | None:
    if fmt.encoding.lower() not in {"pcm", "pcm_s16le", "pcm16", "raw", "pcm_f32le", "f32le"}:
        return None
    required = (onset_sample + 1) * fmt.channels * fmt.sample_width_bytes
    cumulative = 0
    for chunk in chunks:
        cumulative += len(chunk.audio or b"")
        if cumulative >= required:
            return (chunk.at_ns - started_ns) / 1_000_000
    return None


def _onset_metrics(
    onset: OnsetResult,
    chunks: list[ProviderEvent],
    fmt: Any,
    started_ns: int,
    ttfb_ms: float,
) -> dict[str, float | None]:
    audio_clock = ttfb_ms + onset.onset_ms
    available = _onset_available_ms(chunks, fmt, onset.onset_sample, started_ns)
    canonical = max(audio_clock, available or audio_clock)
    return {
        "ttfa_audio_clock_ms": audio_clock,
        "ttfa_available_ms": available,
        "ttfa_ms": canonical,
    }


def _onset_metadata(onset: OnsetResult, metrics: dict[str, float | None]) -> dict[str, Any]:
    return {
        "onset_sample": onset.onset_sample,
        "onset_ms": onset.onset_ms,
        "threshold": onset.threshold,
        "low_threshold": onset.low_threshold,
        "noise_floor": onset.noise_floor,
        "speech_level": onset.speech_level,
        **metrics,
    }


def _error_code(error: Exception) -> str:
    message = str(error).lower()
    if "timeout" in message:
        return "timeout"
    if "without audio" in message or "empty" in message:
        return "no_audio"
    if "onset" in message:
        return "no_voice_onset"
    if "decode" in message or "ffmpeg" in message:
        return "decode_error"
    return "provider_error"
