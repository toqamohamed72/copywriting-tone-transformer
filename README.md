# Automated Copywriting & Tone Transformer

Turn a raw product description into platform-ready marketing copy. You choose the
**platform** (LinkedIn, Instagram, Email), the **tone**, and the inference settings
(**Temperature**, **Top-P**). Your code, not the end user, compiles the prompt.

DecodeLabs · Generative AI · Project 2 (Batch 2026)

```
python run.py generate --product "NovaDesk" \
  --description "An electric standing desk with memory presets and a quiet dual motor." \
  --platform email --tone luxury --temperature 0.4 --top-p 0.9
```

## What it does

| Capability | Where |
|---|---|
| Inputs: product name, description, platform, tone, Temperature, Top-P | `copywriter/schemas.py` (`CopyRequest`) |
| Input sanitization (control chars, delimiter injection, length limits) | `schemas.sanitize_text` |
| Dynamic prompt templates with variable injection | `copywriter/prompts.py` |
| Platform rules and tone voices as data | `copywriter/specs.py` |
| Model-aware Temperature/Top-P handling | `copywriter/capabilities.py`, `copywriter/llm.py` |
| Structured JSON output validated by Pydantic, with one automatic repair retry | `schemas.py`, `generator.py` |
| Platform length / hashtag validation | `copywriter/validation.py` |
| Real-time async pipeline (`asyncio.gather` + `Semaphore`) | `CopyGenerator.agenerate_many` |
| Retries with randomized exponential backoff (Tenacity) | `llm._retry_kwargs` |
| Bulk pipeline through the OpenAI Batch API (`openbatch` + native fallback) | `copywriter/bulk.py` |
| CLI with `argparse` (custom type converters, `+a` prefix action) | `copywriter/cli.py`, `run.py` |
| Streamlit interface with previews, regenerate, copy, export | `app.py` |
| Heuristic quality evaluation | `copywriter/evaluation.py` |
| Temperature x Top-P experiment runner | `copywriter/experiments.py` |
| Usage logging (tokens, latency, parameters) | `copywriter/usage_log.py` |

### Traceability to the PDF

| PDF requirement | Status |
|---|---|
| Python script taking `Product_Name`, `Platform`, `Tone` | Done (CLI + UI + CSV) |
| Inject variables into a dynamic string template | Done (`prompts.py`) |
| Handle `Temperature` and `Top_P` | Done, with per-model rules |
| Define Pydantic models for output schemas | Done (`GeneratedCopy`, strict JSON Schema) |
| Configure the CLI with `argparse` | Done |
| Master template with f-strings, brand safety, platform constraints | Done |
| Dual pipeline: `asyncio.gather` + Semaphores, and `openbatch` for bulk | Done; see *Limitations* for what was and was not tested live |
| Tenacity retry shield | Done |

## Quick start (VS Code)

```bash
# 1. open the folder in VS Code, then open the integrated terminal (Ctrl+`)
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. configure
cp .env.example .env                 # Windows: copy .env.example .env
# edit .env and set OPENAI_API_KEY (and OPENAI_MODEL if needed)

# 3. verify
python run.py doctor                 # configuration check, no API call
python run.py doctor --ping          # one tiny real request

# 4. use it
python run.py generate --product "NovaDesk" --description "..." +a --tone friendly
streamlit run app.py
```

No API key yet? Add `--backend mock` (CLI) or choose *mock* in the app sidebar. The mock
engine returns **placeholder text built from templates, not model output**. It exists so
you can explore the interface and run the tests offline.

### Choosing a model (important)

Temperature and Top-P are the heart of this project, and **not every model accepts
them**. Per OpenAI's [GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model),
reasoning-capable models reject `temperature` / `top_p` unless reasoning effort is `none`.
`copywriter/capabilities.py` encodes this: for `gpt-6-luna` the app sends
`reasoning_effort="none"` together with your sampling values, and for models that never
accept sampling it omits them and tells you (the UI disables the sliders).

Model lineups change quickly; third-party trackers report that several 2024-2025 models
(e.g. `gpt-4o`, `gpt-4.1` families) are scheduled for retirement in early 2027. Check the
[OpenAI models page](https://developers.openai.com/api/docs/models) and set `OPENAI_MODEL`
accordingly. For models the registry does not know, the backend tries the parameters and
automatically retries without them if the API rejects them.

## CLI reference

```bash
python run.py generate  --product P --description D (--platform linkedin|instagram|email|twitter | +a)
                        [--tone professional|friendly|persuasive|creative|luxury|witty]
                        [--temperature 0..2] [--top-p 0..1] [--variations N] [--evaluate] [--json] [--out f.csv|.json|.md]
python run.py prompt    ...same brief...        # print the compiled prompt, no model call
python run.py bulk      --csv data/sample_products.csv [--mode async|batch] [--concurrency 5] [--wait]
python run.py batch-status  --batch-id batch_...
python run.py batch-collect --batch-id batch_... --out outputs/batch_results.csv
python run.py experiment ...brief... --temps 0.1,0.7,1.0 --top-ps 0.5,1.0 --repeats 3
python run.py doctor [--ping]
```

`+a` (or `++all-platforms`) generates LinkedIn, Instagram and Email in one go.
Exit codes: `0` ok, `1` some generation failed, `2` invalid arguments, `3` configuration error.

## Project structure

```
copywriting-tone-transformer/
├── app.py                  Streamlit interface
├── run.py                  CLI entry point
├── copywriter/
│   ├── config.py           environment-based settings
│   ├── schemas.py          Pydantic input/output models, sanitization, strict JSON Schema
│   ├── specs.py            platform rules, tone voices, copy renderer
│   ├── prompts.py          master templates and the prompt compiler
│   ├── capabilities.py     which models accept Temperature/Top-P
│   ├── llm.py              OpenAI backend (retries, adaptation) + mock backend
│   ├── generator.py        generate / repair / async pipeline
│   ├── validation.py       platform validation and normalization
│   ├── bulk.py             CSV loader and Batch API pipeline
│   ├── evaluation.py       heuristic quality scores
│   ├── experiments.py      Temperature x Top-P sweeps
│   ├── exports.py          CSV / JSON / Markdown export
│   ├── usage_log.py        JSONL usage log
│   └── cli.py              argparse interface
├── data/sample_products.csv
├── scripts/sampling_demo.py   offline Temperature/Top-P math (no API key)
├── tests/                  100 tests
├── docs/                   architecture, concepts, experiments, testing, interview Q&A, demo script
└── .env.example  requirements.txt  requirements-dev.txt  pytest.ini
```

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite covers schemas, prompt compilation (including injection attempts), model
capability rules, retry/adaptation behaviour with simulated API errors, the async pipeline's
concurrency limit, CSV and Batch API handling with a fake client, evaluation heuristics, the
CLI, and the Streamlit app (driven headlessly with Streamlit's `AppTest`).

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): components, data flow, design decisions
- [`docs/CONCEPTS.md`](docs/CONCEPTS.md): the 11 core concepts, in simple English and Egyptian Arabic
- [`docs/PROMPT_ENGINEERING.md`](docs/PROMPT_ENGINEERING.md): how dynamic orchestration works, with real compiled prompts
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): Temperature / Top-P protocol, offline math demo, results template
- [`docs/TESTING.md`](docs/TESTING.md): test strategy and evaluation rubric
- [`docs/INTERVIEW_QA.md`](docs/INTERVIEW_QA.md): questions and model answers
- [`docs/DEMO_AND_PRESENTATION.md`](docs/DEMO_AND_PRESENTATION.md): demo scenarios and slide outline

## Tested with

Python 3.12 · openai 3.46 and 3.49 · pydantic 2.14 · tenacity 9.2 · streamlit 1.67 · openbatch 0.0.4
(100 tests pass in a fresh virtual environment built from `requirements-dev.txt`). The real SDK clients,
Tenacity retries and error translation were also exercised end to end with no network access, which
produced the expected "could not reach the API" outcome on both the sync and async paths.

