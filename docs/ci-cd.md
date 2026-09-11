# CI / CD Design

This document explains **why** the CI pipeline is the way it is, and what CD will require before it can exist. It is the design companion to `.github/workflows/ci.yml` and the CI/CD section of the README.

## Scope

- **Present:** one GitHub Actions workflow (`.github/workflows/ci.yml`) doing repository-structure, Python, frontend, and Docker checks.
- **Absent right now (deliberately):** deployment (CD), image publishing to a registry, Kubernetes, Kafka, gRPC, vector DBs, RAG, Redis, multi-agent systems, and any hosting-specific infra.
- **Rule:** nothing is added until a concrete requirement shows up. Deployment exists only once there is a real hosting target to deploy to — see the checklist in [docs/roadmap.md](docs/roadmap.md).

## Why CI Is Present Today

The repository contains documentation only, so the CI value here is **structural**: it enforces the repository contract on every push and every pull request. It verifies that required files exist, are non-empty, and that the repository remains self-consistent as it grows. The design deliberately makes the docs the *first-class* CI citizens, because they are currently the only product of this repo.

## The Single Workflow: ci.yml

```text
PR / push to main
        ↓
   structure       always runs (docs must exist and be non-empty)
   python          activates when Python project files appear
   frontend        activates when package.json appears
   docker          activates when a Dockerfile appears
        ↓
     Merge gate
```

Two intentional properties:

1. **Jobs activate when their component appears, and are *skipped* (never *failed*) until then.** The repo can start out green with a single `structure` job and grow without editing the workflow.
2. **Failure handling is explicit**: `set -euo pipefail` everywhere, no hidden success, and no workflow that "passes" while silently ignoring a broken component. Skipped ≠ failed, and skipped ≠ broken — a skipped job means the component does not exist yet, and that is the honest state today.

## Component Detection

Detection is done by a checkout-based `detect` job, which emits component and lockfile booleans to downstream jobs. A locate step then resolves a sub-directory (`.` by default) so the workflow does not assume everything lives at the repository root. This is necessary because GitHub Actions cannot use `hashFiles()` in job-level conditions before a runner has checked out the repository.

| Job | Activates when | Runs |
| --- | --- | --- |
| `structure` | (always) | docs present and non-empty |
| `python` | `pyproject.toml`, `setup.*`, `requirements*.txt`, `uv.lock`, `Pipfile`, `pytest.*`, etc. | deps install → ruff lint → `ruff format --check` → pytest |
| `frontend` | `package.json` | install → lint → typecheck → build |
| `docker` | a `Dockerfile` | clean-environment image build |

## Python in CI: uv-first, pip fallback

ClauseGuard's training code is expected to be Python. The workflow prefers the repository's actual tooling:

- If `uv.lock` exists → `uv sync --frozen` (uses the committed lockfile, fast and reproducible).
- Otherwise → pip + `requirements*.txt` / `pyproject.toml` editable installs.

Lint/format use **ruff** only when ruff configuration is present (`ruff.toml`, `.ruff.toml`, or `[tool.ruff]` in `pyproject.toml`); otherwise those steps are skipped with an explicit message (no silent "pass", no made-up config).

Tests run only when tests are discovered (`tests/` or `test_*.py` / `*_test.py` files), and the step reports **no tests found** otherwise. The step never fails because tests don't exist yet.

## ML Training Is NOT a PR Check

Full model training (fine-tuning a transformer on the LEDGAR/LexGLUE data) is an **experiment** that does not belong in normal CI — it is slow, expensive, and not something every PR should trigger. The repo distinguishes:

- **CI** (normal jobs): structure, lint, format, tests, builds — fast, per-PR.
- **ML experiment workflow** (`ml-training.yml`, planned, manually triggered): a separate path that runs only when explicitly invoked; it does NOT run on every PR.

See [docs/ml-training.md](docs/ml-training.md) and [docs/ml-pipeline.md](docs/ml-pipeline.md).

## Security

- `permissions: contents: read` — no unnecessary write access, no workflow token escalation.
- No secrets are used anywhere in CI today. No deployment keys exist because there is nothing to deploy to.
- Actions are pinned to well-known major tags (`actions/checkout@v4`, `actions/setup-python@v5`, `actions/setup-node@v4`). Committing action code is not required; pinning major versions is a sensible and standard baseline.

## Why CD Is Absent (and the CD Checklist)

CD deploys the application somewhere. There is **no confirmed hosting target** and **no application code yet**, so a CD pipeline would be fabricated infrastructure — exactly what the project refuses to build.

Future CD will be enabled only when all of these hold (tracked in [docs/roadmap.md](docs/roadmap.md)):

1. A hosting target is confirmed (e.g., a container registry + a deployment platform or VM).
2. Repository secrets are configured for that target (no secrets can be committed).
3. A deployable artifact exists (Docker image of the inference service containing `model.onnx`; backend; frontend).
4. A separate `cd.yml` workflow is added and gated to `main`, running **only after CI passes**.

Until items (1)–(3) are decided, CD stays off — and that is the correct state, not a missing feature.

## Reproducing Locally

Everything CI does should be runnable by hand in the same order:

```bash
# Python (all steps must be in the component directory)
python -m pip install -r requirements.txt   # or: uv sync
ruff check .
ruff format --check .
pytest

# Frontend (in the directory containing package.json)
npm ci          # or pnpm install / yarn install --immutable
npm run lint
npm run typecheck
npm run build

# Docker
docker build .   # or: docker compose build
```

A status note: the CI badge in the README becomes live once the workflow is actually pushed and GitHub has processed it; the workflow cannot be fully exercised from a local environment. Local YAML syntax can be validated, but wait — the job won't run until the config files appear. Locally we validated YAML parseability and cross-referenced every step/output reference; execution semantics (skipped vs failed, activation) surface on GitHub itself.
