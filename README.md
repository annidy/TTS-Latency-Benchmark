# TTS Latency Benchmark

🏆 [**Leaderboard**](https://breezeblue.ai/benchmarks/tts-latency)

This repository contains the self-contained local evaluation code for comparing
end-to-end text-to-speech startup latency across providers.

It measures:

1. **Time to First Byte (TTFB)**: request start to the first non-empty audio chunk.
2. **Initial Silence**: decoded audio duration before the first detected voice onset.
3. **Time to First Audio (TTFA)**: when the first audible speech is available to the
   client.

The same runner lifecycle is used for every model. Provider-specific authentication,
request framing, audio formats, completion events, and connection-reuse rules remain
inside model adapters.

## Supported Models

| Model | HTTP | WebSocket | Standard benchmark protocol |
|---|:---:|:---:|---|
| BreezeBlue Breeze TTS 2 | ✅ | ✅ | WebSocket |
| ElevenLabs Flash v2.5 | ✅ | ✅ | WebSocket |
| ElevenLabs v3 | ✅ | ❌ | HTTP |
| Fish Audio S2.1 Pro | ✅ | ✅ | HTTP |
| Cartesia Sonic 3.5 | ✅ | ✅ | WebSocket |
| Inworld TTS-2 | ✅ | ✅ | WebSocket |
| xAI TTS | ✅ | ✅ | WebSocket |
| Async Flash 1.5 | ✅ | ✅ | WebSocket |
| Speechify Simba 3.2 | ✅ | ❌ | HTTP |

<sub>Fish Audio uses HTTP in the standard benchmark because its WebSocket protocol adds approximately 300 ms of initial silence after the first turn in a persistent session.</sub>

## Setup

Python 3.11 or newer is required.

```bash
git clone https://github.com/breezeblue-ai/TTS-Latency-Benchmark.git
cd TTS-Latency-Benchmark

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`ffmpeg` is an optional runtime dependency. It is needed only when a selected provider
returns compressed audio such as MP3, which must be decoded to PCM for voice-onset
detection. Providers returning PCM or supported WAV audio do not require it.

## Credentials

Copy the environment template:

```bash
cp .env.example .env.local
```

Fill in the API key and voice ID for each model you want to run. Credentials for unused
models may remain empty.

```dotenv
TTFA_CLIENT_REGION=local
BREEZEBLUE_API_KEY=your_api_key
BREEZEBLUE_VOICE_ID=your_voice_id
```

API keys are intentionally omitted from this repository. Do not commit `.env.local` or
put secrets directly into suite and catalog YAML files.

## Validate The Installation

List the supported public model IDs:

```bash
python -m benchmark models
```

Validate one model's configuration, credentials, and Python runtime without sending a
synthesis request. The diagnostic output also reports whether the optional `ffmpeg`
executable is available:

```bash
python -m benchmark doctor \
  --config builtin:smoke/breezeblue-breeze-tts-2.yaml \
  --env-file .env.local
```

## Run The Standard Benchmark

The standard local benchmark uses 40 shared English prompts selected from the Seed-TTS
evaluation metadata. Every model receives the same prompts in the same order.

```bash
python -m benchmark run \
  --config builtin:suites/standard-seedtts-en-40.yaml \
  --env-file .env.local
```

The preset applies the following execution policy:

- three excluded warm-up requests per model;
- one measured request for each of the 40 prompts;
- requests for the same model are strictly serial;
- different models run concurrently;
- one warmed WebSocket per model when selected;
- one warmed persistent HTTP connection pool per HTTP model.

The implementation uses `iteration_order: interleaved` and
`provider_execution: parallel`. For each text, it starts at most one request per model
and waits for every model to finish before advancing to the next text. This preserves
serial request order within a model while allowing different models to run in parallel.

The default protocol for each model is declared explicitly in
`benchmark/presets/suites/standard-seedtts-en-40.yaml`.

### Dataset Source

The 40 English prompts in
[`benchmark/presets/texts/seedtts-en-40.yaml`](benchmark/presets/texts/seedtts-en-40.yaml)
were selected from the English evaluation metadata released by the
[Seed-TTS evaluation project](https://github.com/BytedanceSpeech/seed-tts-eval). The
Seed-TTS maintainers state that the English evaluation samples were extracted from
[Mozilla Common Voice](https://commonvoice.mozilla.org/), whose text contributions and
datasets are made available under the
[Creative Commons CC0 1.0 public-domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).

The checked-in set was produced by reading the target-text column from
`seedtts_testset/en/meta.lst`, retaining the first occurrence of each distinct text,
sorting by character count in descending order with stable source-row tie-breaking, and
selecting the first 40 entries. The repository redistributes only these text prompts; it
does not include Common Voice or Seed-TTS audio, speaker metadata, or reference
recordings.

## Run One Model

Every model has a smoke configuration under `benchmark/presets/smoke/`. A smoke test
runs all transports declared for that model:

```bash
python -m benchmark run \
  --config builtin:smoke/breezeblue-breeze-tts-2.yaml \
  --env-file .env.local
```

Available smoke configurations:

```text
builtin:smoke/async-flash-v1.5.yaml
builtin:smoke/breezeblue-breeze-tts-2.yaml
builtin:smoke/cartesia-sonic-3.5.yaml
builtin:smoke/elevenlabs-flash-v2.5.yaml
builtin:smoke/elevenlabs-v3.yaml
builtin:smoke/fish-audio-s2.1-pro.yaml
builtin:smoke/inworld-tts-2.yaml
builtin:smoke/speechify-simba-3.2.yaml
builtin:smoke/xai-tts.yaml
```

## Run Protocol-specific Diagnostics

The following suites are useful for protocol diagnostics; they are not the standard
mixed-protocol benchmark. Run all supported models over warmed HTTP connections:

```bash
python -m benchmark run \
  --config builtin:suites/diagnostic-all-http.yaml \
  --env-file .env.local
```

Run the WebSocket diagnostic suite:

```bash
python -m benchmark run \
  --config builtin:suites/diagnostic-all-websocket.yaml \
  --env-file .env.local
```

These two diagnostic suites execute models serially. The standard
`standard-seedtts-en-40.yaml` suite executes different models in parallel as described
above. Persistent HTTP pools and WebSocket connections are prepared and warmed before
measured requests begin. Warm-up samples are excluded from the reported distribution.

## Outputs

By default, each run creates a timestamped directory below `outputs/runs/` containing:

```text
config.redacted.yaml
manifest.json
samples.jsonl
summary.json
summary.csv
report.md
dumbbells/
```

`samples.jsonl` contains per-request observations and event timing. The summary files
contain aggregated percentiles. `report.md` contains the result table and a horizontal
dumbbell chart showing TTFB p50, TTFA p50, and TTFA p95.

Use `--save-audio` to retain successful generated audio and `--boxplots` to generate
optional TTFB, initial-silence, and TTFA box plots:

```bash
python -m benchmark run \
  --config builtin:smoke/breezeblue-breeze-tts-2.yaml \
  --env-file .env.local \
  --output-dir outputs/my-run \
  --save-audio \
  --boxplots
```

Generated audio may contain sensitive text or voice characteristics. Handle saved
artifacts accordingly.

## Measurement Protocol

Timing uses a monotonic high-resolution client clock.

- TTFB ignores provider metadata, acknowledgements, and context messages that do not
  contain audio.
- Initial silence is calculated after decoding the complete audio. Raw samples retain
  both the v1 absolute/density detector and v2 relative dual-energy hysteresis detector.
- TTFA combines audio arrival timing with the detected voice-onset position.
- Total latency ends at the provider-specific completion event or HTTP response EOF.
- RTF is total latency divided by generated audio duration.

For a persistent profile, DNS, TCP, TLS, and WebSocket setup occur before the measured
request. For a profile configured with a new connection, setup time is included.

## Custom Local Suite

Built-in configuration is organized as:

```text
benchmark/presets/
├── catalog/   provider endpoints, model IDs, voices, audio, and protocol rules
├── smoke/     one runnable configuration per model
├── suites/    local multi-model configurations
└── texts/     reusable text inputs
```

Create a separate suite YAML to change models, transport, text, warm-up count,
repetitions, or model concurrency:

```yaml
schema_version: 2
suite:
  id: local-breeze-test
  description: Local WebSocket test
catalog: [builtin:catalog/breezeblue.yaml]
targets:
  - model: breezeblue.breeze-tts-2
    transports: [websocket]
execution:
  profiles:
    - id: warm-ws
      connection_mode: persistent
      iteration_order: interleaved
      provider_execution: parallel
      warmup_runs: 3
      repetitions: 10
texts:
  ref: builtin:texts/smoke-en.yaml
clients:
  - id: local
    region_env: TTFA_CLIENT_REGION
    runner: {kind: local}
outputs:
  base_dir: outputs/runs
```

Run it with:

```bash
python -m benchmark run --config local-suite.yaml --env-file .env.local
```

Set `provider_execution: serial` to run different models one at a time. Keep
`iteration_order: interleaved` with `provider_execution: parallel` when each model must
remain serial internally while different models execute concurrently.
