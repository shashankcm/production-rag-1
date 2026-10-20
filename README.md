# production-rag-1: a hardened LLM API

A FastAPI service that wraps a LangGraph LLM agent in the production concerns that sit around a model call: rate limiting, prompt-injection and PII checks, a response cache, retry with a fallback model, output validation, structured logs and metrics.

**What this is, and is not.** It is a reference implementation of the API layer. It does **not** retrieve documents yet, so despite the repository name there is no RAG step: the agent calls the model directly. See [Known limitations](#known-limitations).

## Request flow

```
Client request
  -> rate limiter                (slowapi, per client address, default 20/minute)
  -> security: input check       (prompt-injection patterns -> 400; PII masked before the model sees it)
  -> cache lookup                (hit: return the stored answer, skip the model)
  -> LangGraph agent             (primary model; if it fails, the fallback model, retried until MAX_RETRIES; then an error -> 503)
  -> security: output check      (PII masked; harmful patterns replace the whole answer)
  -> cache store
  -> metrics and JSON logs
  -> JSON response
```

The cache stores the **validated** answer, so a blocked or masked answer is never served in raw form from the cache. If every model call fails the caller receives a generic 503, nothing is cached, and the full error is logged server-side only. An unexpected exception returns a generic 500 the same way.

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/chat` | `{"message": "...", "thread_id": "..."}` returns the answer, `model_used`, `cached`, timing and `security_notes` |
| GET | `/health` | Reports whether the agent, security pipeline and cache are initialised |
| GET | `/metrics` | Request, error, latency, token and cache-hit summary |
| GET | `/cache/stats` | Cache hits, misses, size and hit rate |

## Configuration

Read from environment variables or a `.env` file (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | none | Required to call the models |
| `PRIMARY_MODEL`, `FALLBACK_MODEL` | `gpt-4o-mini` | Models used by the agent |
| `MAX_RETRIES` | `3` | Retries before the agent gives up |
| `RATE_LIMIT` | `20/minute` | slowapi limit string for `/chat` |
| `CACHE_TTL_SECONDS` | `300` | How long a cached answer is served |
| `APP_ENV`, `LOG_LEVEL` | `development`, `INFO` | Environment label and log level |
| `LANGCHAIN_*` | tracing on | Optional LangSmith tracing |

## Run it

```bash
uv sync
cp .env.example .env            # then add your OPENAI_API_KEY
uv run uvicorn app.main:app --port 8000
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' -d '{"message": "What is a pod?"}'
```

With Docker: `docker compose up --build` (the image runs as a non-root user and has a health check).
On Render: `render.yml` defines a free web service. The two secrets use `sync: false`, so Render asks for them in its dashboard.

## Tests

```bash
uv run pytest
```

56 tests, no API key or network needed (the models are replaced by fakes):

- **Cache:** lookup, key normalisation, TTL expiry with a controllable clock, statistics.
- **Security:** nine injection patterns, five PII types, output blocking, and the pipeline.
- **Agent:** primary path, fallback path, a total failure that is flagged as an error and never shown as an answer, and a stop after `MAX_RETRIES`.
- **API:** cache hit on a repeated question, blocked injection (the agent is never called), PII masked before the agent sees it, harmful output replaced, 422 on an empty message, the 21st request in a minute returns 429, a model failure that returns 503 and is not cached, and a 500 that does not leak the internal error.

Two tests are strict expected-failures that document known over-blocking (below).

## Bugs the tests found

Writing the tests turned up real defects in the original code, now fixed:

- **The fallback retry crashed.** The routing function could return `"fallback"` to retry, but the graph had no edge for it, so the second fallback failure raised `KeyError: 'fallback'` with the default `MAX_RETRIES=3`.
- **A total failure was returned as an answer.** The error handler wrapped the exception text as a normal model message and cleared the error flag, so the API answered 200 with internal error text in the body, and cached it. (Reachable when `MAX_RETRIES` is 1 or 2; with the default of 3 the crash above happened first.)
- **500 responses leaked internals.** The handler returned `str(exception)` to the caller.
- **`render.yml` set the API keys to the string `"false"`** (`value: false`) instead of prompting for them.
- Two stray auto-imported lines (`pydantic_settings.main.T`, `slowapi.extension.Dict`) were removed.

## Known limitations

These are real and were found by reading the code and writing the tests:

- **No retrieval.** There is no vector store or document step. The `thread_id` in the request is accepted but not passed to the agent, so there is no conversation memory.
- **Over-blocking.** The input pattern `system prompt` rejects legitimate questions about system prompts, and the output pattern `api key` replaces any answer that mentions an API key, even a harmless one. Both are recorded as strict `xfail` tests.
- **In-memory state.** The cache and the metrics live in one process, so they reset on restart and are not shared across instances. A real deployment would use Redis and Prometheus.
- **Metrics skew.** Blocked and failed requests are counted as cache misses, which lowers the reported cache hit rate.
- **Pattern-based security.** Regular expressions catch known phrasings, not novel attacks, and the PII patterns are US-centric.
- **Rate limit configured at import time**, per client address.
- `langchain-anthropic` is a dependency and `ANTHROPIC_API_KEY` is in `.env.example`, but the agent only uses OpenAI models.
