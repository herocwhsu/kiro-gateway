# Codex Responses API Adapter Design

## Goal

Add a native, authenticated OpenAI Responses API compatibility endpoint at `POST /v1/responses` so Codex CLI can use Kiro-provided `gpt-*` model IDs through `kiro-gateway`. The existing OpenAI Chat Completions and Anthropic Messages integrations, especially Claude Code's `POST /v1/messages` path, must retain their current behavior.

## Context and constraints

Codex CLI 0.144.4 custom providers use `wire_api = "responses"`; it cannot use this gateway's existing `/v1/chat/completions` endpoint directly. The gateway already translates OpenAI Chat Completions requests into Kiro's `generateAssistantResponse` workflow and deliberately passes unknown model IDs through to Kiro. Therefore an exact Kiro CLI `gpt-*` model ID may be supplied to Responses even when absent from `/v1/models`.

The running production gateway remains bound to `127.0.0.1:8000`. Source changes in this isolated worktree do not affect it. Replacing that container later is a separate, explicit approval step because it briefly interrupts Claude Code traffic.

The clean `fc8aec4` baseline has 1,671 passing and 29 failing tests before this work: 27 Anthropic failures from a missing `hidden_models` argument in `converters_anthropic.py`, plus two HTTP-client selection assertions. This feature must neither change nor mask those failures.

## Selected approach

Implement an in-gateway adapter rather than a separate proxy/container. The endpoint will translate the supported Responses request subset into the existing `ChatCompletionRequest` model, invoke the existing chat-completions execution path, then translate the JSON response or SSE events back into Responses format.

This preserves the established Kiro payload conversion, account selection/failover, retry behavior, token handling, debug logging, and Kiro model pass-through. It also confines the change to a new API surface and router registration, protecting `/v1/messages` and `/v1/chat/completions` semantics.

Alternatives rejected:

1. A separate adapter service would duplicate authentication, networking, Kiro account selection, and deployment concerns.
2. Adding a Chat Completions wire option to Codex is not viable because the installed Codex CLI custom-provider interface supports Responses only.

## Architecture

```text
Codex CLI
  │ POST /v1/responses
  ▼
Responses router (existing Bearer-token verification)
  ├─ restore bounded in-memory conversation state by `previous_response_id`
  ├─ Responses input/tools → ChatCompletionRequest
  ├─ invoke existing `chat_completions` route function
  ├─ Chat JSON/SSE → Responses JSON/SSE
  └─ save a response snapshot for a later continuation
        ▼
Existing Kiro Chat Completions pipeline
        ▼
Kiro `generateAssistantResponse`
```

### New modules

- `kiro/models_responses.py`: focused Pydantic request models for the supported Responses subset and response/item helpers.
- `kiro/responses_adapter.py`: pure conversion functions, SSE event conversion, response construction, and the bounded in-memory state store. It must not depend on FastAPI application state.
- `kiro/routes_responses.py`: `POST /v1/responses`, authentication dependency, state lookup/store orchestration, and delegation to the existing chat route.
- `tests/unit/test_responses_adapter.py`: unit tests for request/input/tool conversion, response conversion, SSE event order, and state-store expiry/eviction.
- `tests/unit/test_routes_responses.py`: route tests for authentication, JSON and streaming output, continuation, and error translation.

`main.py` only imports and registers the new router alongside `openai_router` and `anthropic_router`.

## Request compatibility

The first version supports the parts of Responses used by the Codex agent workflow:

- `model` is required and is passed through unchanged, including exact Kiro CLI `gpt-*` IDs.
- `stream` defaults to `false`.
- `instructions` becomes a leading Chat `system` message.
- `input` accepts a string or an ordered list of supported items.
  - `input_text` becomes a Chat `user` message.
  - message items with `role` in `developer`, `system`, `user`, or `assistant` become corresponding Chat messages. A message `content` value may be a string or an ordered array containing only `input_text` and `output_text` parts; their text is concatenated in order. `developer` maps to `system`, because the gateway's current Chat model accepts system messages and Kiro has a single system-prompt channel.
  - `output_text` in an assistant message becomes Chat assistant text.
  - `function_call` becomes an assistant `tool_calls` entry, preserving its call ID, function name, and JSON arguments string.
  - `function_call_output` becomes a Chat `tool` message using `call_id` as `tool_call_id`; string output is used directly and structured output is JSON-serialized.
- OpenAI function tools (`{"type":"function", "name":..., "description":..., "parameters":...}`) map to the existing Chat tool shape (`{"type":"function", "function": {...}}`).
- `tool_choice`, `temperature`, `top_p`, `max_output_tokens`, and `reasoning.effort`, when present in the supported request, map to their closest existing Chat request fields. `max_output_tokens` maps to `max_completion_tokens`.
- Unimplemented Responses-specific modalities or item types receive a deterministic `400` error that names the unsupported type. Unknown extra request fields remain tolerated to avoid rejecting harmless Codex metadata.

## Conversation state and continuation

Responses is stateless at the Kiro layer, so the adapter maintains response snapshots in process memory.

A snapshot records: response ID, creation time, model, accumulated Chat message history, and the prior response output items. After each successful non-streaming response, and after a streaming response has completed, the adapter stores the request's effective history followed by the assistant message reconstructed from the converted Chat response.

For a request with `previous_response_id`, the router:

1. Retrieves its snapshot.
2. Uses the snapshot history as the prefix.
3. Adds the new request's instructions and input.
4. Requires any explicitly supplied model to equal the stored model; otherwise returns `400` rather than silently mixing models in a conversation.

State is bounded to 100 entries and a two-hour TTL. Expired entries are removed on access/store; least-recently-created entries are evicted when the maximum is exceeded. Unknown or expired IDs return `404` in OpenAI-style error JSON. Gateway restart clears all snapshots by design.

A continuation after a function call is represented by the stored assistant tool call plus an incoming `function_call_output`, so the existing Kiro conversion receives the same assistant-tool/user-tool-result sequence expected by its established tool pipeline.

## Non-streaming response format

A successful request returns an OpenAI Responses-style object:

- `object: "response"`
- opaque `id` with a `resp_` prefix
- `status: "completed"`
- request model
- ordered `output` items
- `output_text` assembled from returned assistant text parts

Chat assistant text converts to a Responses `message` output item containing `output_text`. Each Chat tool call converts to a `function_call` output item with its stable call ID, name, and arguments. If Chat response metadata includes token usage, it is mapped to the Responses `usage` fields where source values are available; unavailable detailed fields are omitted rather than invented.

When the chat route returns an error response, the Responses router preserves its status code and returns an OpenAI-compatible `error` object. Request validation failures use FastAPI's existing validation behavior; adapter conversion failures use explicit `400` errors.

## Streaming response format

When `stream: true`, the router returns `text/event-stream` and emits a minimal, ordered Responses event sequence compatible with Codex:

1. `response.created`
2. `response.in_progress`
3. zero or more output-item/content-part add and text-delta events
4. function-call argument-delta and done events where applicable
5. `response.output_item.done` for each completed output item
6. `response.completed`

The converter parses only the gateway's own Chat Completion SSE events. It assigns output indexes deterministically, aggregates deltas to reconstruct final assistant text and function-call arguments, and stores the completed snapshot only after the terminal completion event. On an upstream stream failure it emits `response.failed`, does not persist incomplete state, and closes underlying resources through the existing chat stream wrapper.

## Isolation and compatibility safeguards

- `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`, and `/v1/chat/completions` are not modified semantically.
- The Responses router uses the existing `verify_api_key`, so it accepts only `Authorization: Bearer <PROXY_API_KEY>`, exactly as the OpenAI-compatible route does.
- No model allowlist is introduced. Direct `gpt-*` IDs continue through the current optimistic Kiro model resolver.
- No response state, model IDs, or proxy secrets are written to disk or committed.
- Tests use the existing global network blocker; no unit test contacts Kiro or any external API.

## Testing and validation

TDD will add a focused failing test before each production behavior. Coverage must include:

1. text input, instructions, all supported message roles, and exact GPT model-ID pass-through;
2. `input_text`, `output_text`, function tools, function calls, and `function_call_output` conversions;
3. JSON response shape and assistant text/tool-call output items;
4. required Bearer authentication and no effect on the Anthropic route;
5. `previous_response_id` history restoration, tool-result continuation, unknown ID `404`, TTL expiry, capacity eviction, and model mismatch `400`;
6. streamed text and tool-call events, terminal completion, and no state persistence after failed streams;
7. full-suite comparison against the pre-existing 29 failures, with no additional unrelated regressions.

A separate loopback-only test container on `127.0.0.1:8001` may be used for end-to-end validation after unit coverage. The existing production container on `127.0.0.1:8000` must not be rebuilt or replaced without explicit approval.

## Codex configuration after deployment

After the endpoint is deployed and verified, Codex can use an external profile similar to:

```toml
model = "<exact gpt-* ID selected by Kiro CLI /model>"
model_provider = "kiro"

[model_providers.kiro]
name = "Kiro Gateway"
base_url = "http://127.0.0.1:8000/v1"
env_key = "KIRO_GATEWAY_API_KEY"
wire_api = "responses"
supports_websockets = false
```

The gateway API key is supplied only through the environment variable and is never placed in this file, source code, test fixtures, logs, or commits.

## Self-review

This design is a single focused adapter feature, not a separate subsystem. It explicitly fixes the support boundary for the first release, defines model/state/error behavior, names all created and changed files, and preserves existing Claude and Chat Completions route semantics. It contains no placeholders or deferred design decisions.