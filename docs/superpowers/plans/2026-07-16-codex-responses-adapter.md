# Codex Responses API Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /v1/responses` so Codex CLI can use Kiro-provided `gpt-*` models without changing existing Claude Code or Chat Completions behavior.

**Architecture:** A new Responses router will convert supported Responses requests to the existing `ChatCompletionRequest`, invoke the existing `chat_completions` Kiro path, then convert JSON/SSE output back to Responses objects/events. A bounded `ResponseStateStore` is scoped to `app.state` and retains Chat history for `previous_response_id` continuation.

**Tech Stack:** Python 3.10, FastAPI, Pydantic v2, pytest/pytest-asyncio, existing `StreamingResponse` and Chat Completions converter.

---

## File structure

| Path | Responsibility |
|---|---|
| `kiro/models_responses.py` | Lenient Pydantic contract for supported Responses requests and response item helpers. |
| `kiro/responses_adapter.py` | Pure request/tool conversion, Chat JSON/SSE conversion, and TTL/capacity-bounded state store. |
| `kiro/routes_responses.py` | Authenticated endpoint, state orchestration, and delegation to `chat_completions`. |
| `main.py` | Create one state store during lifespan and register the router. |
| `tests/unit/test_models_responses.py` | Request contract tests. |
| `tests/unit/test_responses_adapter.py` | Conversion, state, output, and SSE tests without FastAPI. |
| `tests/unit/test_routes_responses.py` | Endpoint/auth/continuation/streaming integration tests with the existing global network block. |
| `tests/unit/test_main_routes.py` | Router registration and existing-route preservation test. |
| `README.md` | A documented Codex profile template using `wire_api = "responses"` with no secret. |

Existing `kiro/routes_openai.py`, `kiro/routes_anthropic.py`, `kiro/converters_openai.py`, and `kiro/converters_anthropic.py` are not changed.

### Task 1: Define the Responses request contract

**Files:**
- Create: `tests/unit/test_models_responses.py`
- Create: `kiro/models_responses.py`

- [ ] **Step 1: Write failing contract tests**

```python
import pytest
from pydantic import ValidationError
from kiro.models_responses import ResponsesRequest

def test_accepts_codex_text_request_and_unknown_metadata():
    request = ResponsesRequest.model_validate({
        "model": "gpt-5",
        "instructions": "Be concise.",
        "input": "Hello",
        "max_output_tokens": 200,
        "reasoning": {"effort": "high"},
        "metadata": {"codex": "preserved"},
    })
    assert request.model == "gpt-5"
    assert request.input == "Hello"
    assert request.reasoning_effort == "high"
    assert request.model_extra["metadata"] == {"codex": "preserved"}

def test_requires_model():
    with pytest.raises(ValidationError):
        ResponsesRequest.model_validate({"input": "Hello"})

def test_accepts_message_and_function_items():
    request = ResponsesRequest.model_validate({
        "model": "gpt-5",
        "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": "Rules"}]},
            {"type": "function_call_output", "call_id": "call_1", "output": "done"},
        ],
        "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}],
    })
    assert request.input[0]["role"] == "developer"
    assert request.tools[0].name == "read_file"
```

- [ ] **Step 2: Run the contract test**

Run: `python3 -m pytest tests/unit/test_models_responses.py -q`
Expected: collection failure because `kiro.models_responses` does not yet exist.

- [ ] **Step 3: Add minimal Pydantic models**

```python
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel

ResponsesInput = Union[str, List[Dict[str, Any]]]

class ResponsesFunctionTool(BaseModel):
    type: Literal["function"]
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None
    model_config = {"extra": "allow"}

class ResponsesRequest(BaseModel):
    model: str
    input: ResponsesInput
    instructions: Optional[str] = None
    previous_response_id: Optional[str] = None
    stream: bool = False
    tools: Optional[List[ResponsesFunctionTool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    reasoning: Optional[Dict[str, Any]] = None
    model_config = {"extra": "allow"}

    @property
    def reasoning_effort(self) -> Optional[str]:
        return (self.reasoning or {}).get("effort")
```

Keep message items intentionally dictionary-shaped: the adapter, not Pydantic, rejects unsupported `type` values with an actionable `400`.

- [ ] **Step 4: Re-run the contract test**

Run: `python3 -m pytest tests/unit/test_models_responses.py -q`
Expected: all tests pass.

### Task 2: Convert Responses inputs and maintain continuation state

**Files:**
- Create: `tests/unit/test_responses_adapter.py`
- Create: `kiro/responses_adapter.py`

- [ ] **Step 1: Write failing conversion/state tests**

```python
from kiro.models_responses import ResponsesRequest
from kiro.responses_adapter import (
    ResponseStateStore,
    ResponsesConversionError,
    responses_to_chat_request,
)

def test_converts_instructions_messages_tools_and_gpt_model_unchanged():
    request = ResponsesRequest.model_validate({
        "model": "gpt-5",
        "instructions": "System rules",
        "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": "Developer rules"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "Question"}]},
        ],
        "tools": [{"type": "function", "name": "read_file", "description": "Read", "parameters": {"type": "object"}}],
        "max_output_tokens": 123,
        "reasoning": {"effort": "medium"},
    })

    chat_request = responses_to_chat_request(request)

    assert chat_request.model == "gpt-5"
    assert [(message.role, message.content) for message in chat_request.messages] == [
        ("system", "System rules"),
        ("system", "Developer rules"),
        ("user", "Question"),
    ]
    assert chat_request.tools[0].function.name == "read_file"
    assert chat_request.max_completion_tokens == 123
    assert chat_request.reasoning_effort == "medium"

def test_converts_function_call_and_function_call_output():
    request = ResponsesRequest.model_validate({
        "model": "gpt-5",
        "input": [
            {"type": "function_call", "call_id": "call_1", "name": "read_file", "arguments": "{\"path\":\"a.py\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": {"contents": "text"}},
        ],
    })

    chat_request = responses_to_chat_request(request)

    assert chat_request.messages[0].tool_calls[0]["id"] == "call_1"
    assert chat_request.messages[0].tool_calls[0]["function"]["name"] == "read_file"
    assert chat_request.messages[1].role == "tool"
    assert chat_request.messages[1].tool_call_id == "call_1"
    assert chat_request.messages[1].content == '{"contents": "text"}'

def test_rejects_unsupported_input_item_type():
    request = ResponsesRequest.model_validate({
        "model": "gpt-5",
        "input": [{"type": "input_image", "image_url": "https://example.invalid/image.png"}],
    })

    with pytest.raises(ResponsesConversionError, match="input_image"):
        responses_to_chat_request(request)

def test_store_expires_entries_and_evicts_oldest(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("kiro.responses_adapter.time.time", lambda: now[0])
    store = ResponseStateStore(max_entries=1, ttl_seconds=10)

    store.put("resp_old", "gpt-5", [])
    now[0] = 1.0
    store.put("resp_new", "gpt-5", [])

    assert store.get("resp_old") is None
    assert store.get("resp_new").model == "gpt-5"

    now[0] = 12.0
    assert store.get("resp_new") is None
```

- [ ] **Step 2: Run the adapter conversion/state tests**

Run: `python3 -m pytest tests/unit/test_responses_adapter.py -q`
Expected: collection failure because `kiro.responses_adapter` does not exist.

- [ ] **Step 3: Implement conversion and state-store primitives**

Implement these exact public interfaces:

```python
from collections import OrderedDict
import copy
import time
from dataclasses import dataclass

class ResponsesConversionError(ValueError):
    pass

@dataclass
class ResponseSnapshot:
    response_id: str
    model: str
    messages: List[ChatMessage]
    output: List[Dict[str, Any]]
    created_at: float

class ResponseStateStore:
    def __init__(self, max_entries: int = 100, ttl_seconds: int = 7200) -> None:
        self._snapshots: OrderedDict[str, ResponseSnapshot] = OrderedDict()
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds

    def put(self, response_id: str, model: str, messages: List[ChatMessage],
            output: Optional[List[Dict[str, Any]]] = None) -> ResponseSnapshot:
        self._purge_expired()
        snapshot = ResponseSnapshot(response_id, model, copy.deepcopy(messages), output or [], time.time())
        self._snapshots.pop(response_id, None)
        self._snapshots[response_id] = snapshot
        while len(self._snapshots) > self._max_entries:
            self._snapshots.popitem(last=False)
        return copy.deepcopy(snapshot)

    def get(self, response_id: str) -> Optional[ResponseSnapshot]:
        self._purge_expired()
        snapshot = self._snapshots.get(response_id)
        return copy.deepcopy(snapshot) if snapshot else None

    def clear(self) -> None:
        self._snapshots.clear()
```

`put()` must purge expired entries, replace same-ID entries, deep-copy Pydantic messages, and evict the oldest insertion once capacity is exceeded. `get()` must purge expired entries first and return a deep copy so callers cannot mutate stored history.

Implement:

```python
def responses_to_chat_request(
    request: ResponsesRequest,
    history: Optional[List[ChatMessage]] = None,
) -> ChatCompletionRequest:
    messages = [message.model_copy(deep=True) for message in history or []]
    if request.instructions:
        messages.append(ChatMessage(role="system", content=request.instructions))
    messages.extend(convert_responses_input_to_chat_messages(request.input))
    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        stream=request.stream,
        temperature=request.temperature,
        top_p=request.top_p,
        max_completion_tokens=request.max_output_tokens,
        reasoning_effort=request.reasoning_effort,
        tools=convert_responses_tools(request.tools),
        tool_choice=request.tool_choice,
    )
```

`convert_responses_input_to_chat_messages()` turns `instructions` and `developer` messages into `system`, converts `input_text`/`output_text` to ordered text, and rejects empty text input or unsupported item/content-part types with `ResponsesConversionError`. `convert_responses_tools()` creates `Tool(type="function", function=ToolFunction(name=tool.name, description=tool.description, parameters=tool.parameters))`. A `function_call` creates `ChatMessage(role="assistant", content=None, tool_calls=[{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}])`; a `function_call_output` creates `ChatMessage(role="tool", tool_call_id=call_id, content=serialized_output)`.

- [ ] **Step 4: Re-run adapter conversion/state tests**

Run: `python3 -m pytest tests/unit/test_responses_adapter.py -q`
Expected: all conversion, TTL, and eviction tests pass.

### Task 3: Convert Chat completion JSON and SSE to Responses output

**Files:**
- Modify: `tests/unit/test_responses_adapter.py`
- Modify: `kiro/responses_adapter.py`

- [ ] **Step 1: Add failing JSON/SSE tests**

```python
import json

from kiro.responses_adapter import (
    ResponsesStreamAccumulator,
    chat_completion_to_response,
    stream_chat_sse_to_responses,
)

async def iter_async(items):
    for item in items:
        yield item

def event_type(frame):
    data_line = next(line for line in frame.splitlines() if line.startswith("data: "))
    return json.loads(data_line.removeprefix("data: "))["type"]

def test_converts_chat_completion_text_and_tool_calls_to_response_items():
    response, assistant_message = chat_completion_to_response({
        "model": "gpt-5",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "I will inspect it.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"a.py\"}"},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }, "resp_test")

    assert response["id"] == "resp_test"
    assert response["status"] == "completed"
    assert response["output_text"] == "I will inspect it."
    assert [item["type"] for item in response["output"]] == ["message", "function_call"]
    assert response["output"][1]["call_id"] == "call_1"
    assert assistant_message.tool_calls[0]["function"]["name"] == "read_file"

@pytest.mark.asyncio
async def test_streams_text_then_terminal_completion_and_collects_history():
    chat_sse = [
        'data: {"choices":[{"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}\n\n',
        'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n\n',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n',
        "data: [DONE]\n\n",
    ]
    accumulator = ResponsesStreamAccumulator("resp_stream", "gpt-5")

    events = [event async for event in stream_chat_sse_to_responses(iter_async(chat_sse), accumulator)]

    assert [event_type(event) for event in events] == [
        "response.created", "response.in_progress", "response.output_item.added",
        "response.content_part.added", "response.output_text.delta",
        "response.output_text.delta", "response.content_part.done",
        "response.output_item.done", "response.completed",
    ]
    assert accumulator.assistant_message.content == "Hello"
    assert accumulator.completed is True
```

Add a tool-call stream test asserting `response.function_call_arguments.delta`, `response.function_call_arguments.done`, a function-call output item, and the reconstructed assistant `tool_calls`.

- [ ] **Step 2: Run the expanded adapter test file**

Run: `python3 -m pytest tests/unit/test_responses_adapter.py -q`
Expected: failure because JSON/SSE response conversion interfaces are missing.

- [ ] **Step 3: Implement response conversion and SSE translator**

Add these public interfaces:

```python
def chat_completion_to_response(
    chat_response: Dict[str, Any],
    response_id: str,
) -> Tuple[Dict[str, Any], ChatMessage]:
    message = chat_response["choices"][0]["message"]
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    assistant = ChatMessage(role="assistant", content=content, tool_calls=tool_calls or None)
    output = build_responses_output_items(content, tool_calls)
    return build_completed_response(response_id, chat_response["model"], output, chat_response.get("usage")), assistant

@dataclass
class ResponsesStreamAccumulator:
    response_id: str
    model: str
    output: List[Dict[str, Any]] = field(default_factory=list)
    text: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None
    completed: bool = False

    @property
    def assistant_message(self) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            content=self.text,
            tool_calls=self.tool_calls or None,
        )

async def stream_chat_sse_to_responses(
    chat_chunks: AsyncIterator[str],
    accumulator: ResponsesStreamAccumulator,
) -> AsyncIterator[str]:
    async for chat_frame in chat_chunks:
        for responses_frame in convert_chat_frame_to_responses_events(chat_frame, accumulator):
            yield responses_frame
    for terminal_frame in complete_responses_stream(accumulator):
        yield terminal_frame
```

Emit each SSE frame as:

```python
def format_sse(event_type: str, data: Dict[str, Any]) -> str:
    payload = {"type": event_type, **data}
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
```

Define the private helpers used by these public functions in the same module: `build_responses_output_items(content, tool_calls)` returns one assistant message item followed by one function-call item per Chat tool call; `build_completed_response(response_id, model, output, usage)` adds `object`, `status`, `output_text`, and available token usage; `convert_chat_frame_to_responses_events(frame, accumulator)` parses one Chat `data:` frame and yields ordered Responses deltas; and `complete_responses_stream(accumulator)` yields terminal content-part, output-item, and completed events while setting `completed` true.

For the first Chat delta, emit `response.created` and `response.in_progress`; add a message item and text content part lazily only when text exists. Parse `data:` JSON frames and ignore `[DONE]` after a proper terminal Chat chunk. Convert the final Chat usage keys to available Responses usage fields. At normal completion emit `response.content_part.done`, `response.output_item.done`, and `response.completed`, then set `accumulator.completed = True`. If iteration raises, emit `response.failed`, leave `completed` false, and re-raise so the existing chat stream wrapper preserves its error behavior.

- [ ] **Step 4: Re-run JSON/SSE adapter tests**

Run: `python3 -m pytest tests/unit/test_responses_adapter.py -q`
Expected: all adapter tests pass.

### Task 4: Add the authenticated `POST /v1/responses` route

**Files:**
- Create: `tests/unit/test_routes_responses.py`
- Create: `kiro/routes_responses.py`

- [ ] **Step 1: Write failing route tests**

Use `monkeypatch` to replace `kiro.routes_responses.chat_completions`; do not make an external request.

```python
def test_responses_requires_bearer_authentication(test_client):
    response = test_client.post("/v1/responses", json={"model": "gpt-5", "input": "Hello"})
    assert response.status_code == 401

def test_responses_returns_json_and_passes_exact_gpt_model(
    test_client, valid_proxy_api_key, monkeypatch,
):
    observed = {}

    async def fake_chat_completions(request, request_data):
        observed["request_data"] = request_data
        return JSONResponse({
            "model": request_data.model,
            "choices": [{"message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        })

    monkeypatch.setattr("kiro.routes_responses.chat_completions", fake_chat_completions)
    response = test_client.post(
        "/v1/responses",
        headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        json={"model": "gpt-5", "instructions": "Answer", "input": "Hi"},
    )

    assert response.status_code == 200
    assert observed["request_data"].model == "gpt-5"
    assert response.json()["object"] == "response"
    assert response.json()["output_text"] == "Hello"

def test_previous_response_id_restores_tool_call_and_accepts_tool_output(
    test_client, valid_proxy_api_key, monkeypatch,
):
    delegated_requests = []
    chat_responses = [
        {
            "model": "gpt-5",
            "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\":\"a.py\"}"},
            }]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        {
            "model": "gpt-5",
            "choices": [{"message": {"role": "assistant", "content": "File read."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    ]

    async def fake_chat_completions(request, request_data):
        delegated_requests.append(request_data)
        return JSONResponse(chat_responses.pop(0))

    monkeypatch.setattr("kiro.routes_responses.chat_completions", fake_chat_completions)
    headers = {"Authorization": f"Bearer {valid_proxy_api_key}"}
    first = test_client.post("/v1/responses", headers=headers, json={"model": "gpt-5", "input": "Read a.py"})
    second = test_client.post(
        "/v1/responses",
        headers=headers,
        json={
            "model": "gpt-5",
            "previous_response_id": first.json()["id"],
            "input": [{"type": "function_call_output", "call_id": "call_1", "output": "contents"}],
        },
    )

    assert second.status_code == 200
    assert delegated_requests[1].messages[-2].tool_calls[0]["id"] == "call_1"
    assert delegated_requests[1].messages[-1].role == "tool"
    assert delegated_requests[1].messages[-1].tool_call_id == "call_1"

def test_unknown_previous_response_id_returns_404(test_client, valid_proxy_api_key):
    response = test_client.post(
        "/v1/responses",
        headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        json={"model": "gpt-5", "previous_response_id": "resp_missing", "input": "Continue"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown or expired previous_response_id"

def test_model_mismatch_for_previous_response_returns_400(test_client, valid_proxy_api_key, monkeypatch):
    async def fake_chat_completions(request, request_data):
        return JSONResponse({
            "model": request_data.model,
            "choices": [{"message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    monkeypatch.setattr("kiro.routes_responses.chat_completions", fake_chat_completions)
    headers = {"Authorization": f"Bearer {valid_proxy_api_key}"}
    first = test_client.post("/v1/responses", headers=headers, json={"model": "gpt-5", "input": "Hello"})
    second = test_client.post(
        "/v1/responses",
        headers=headers,
        json={"model": "gpt-4.1", "previous_response_id": first.json()["id"], "input": "Continue"},
    )
    assert second.status_code == 400
    assert second.json()["detail"] == "model must match previous_response_id"

def test_streaming_response_emits_responses_events_and_persists_after_completion(
    test_client, valid_proxy_api_key, monkeypatch,
):
    delegated_requests = []

    async def chat_chunks():
        yield 'data: {"choices":[{"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_chat_completions(request, request_data):
        delegated_requests.append(request_data)
        return StreamingResponse(chat_chunks(), media_type="text/event-stream")

    monkeypatch.setattr("kiro.routes_responses.chat_completions", fake_chat_completions)
    response = test_client.post(
        "/v1/responses",
        headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        json={"model": "gpt-5", "stream": True, "input": "Hello"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: response.completed" in response.text
    created_data = next(
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and '"type": "response.created"' in line
    )
    snapshot = test_client.app.state.responses_store.get(created_data["response"]["id"])
    assert snapshot.messages[-1].content == "Hello"
    assert delegated_requests[0].model == "gpt-5"
```

Also add this error passthrough test:

```python
def test_responses_preserves_existing_chat_error(test_client, valid_proxy_api_key, monkeypatch):
    async def fake_chat_completions(request, request_data):
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "rate limited", "type": "kiro_api_error", "code": 429}},
        )

    monkeypatch.setattr("kiro.routes_responses.chat_completions", fake_chat_completions)
    response = test_client.post(
        "/v1/responses",
        headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        json={"model": "gpt-5", "input": "Hello"},
    )
    assert response.status_code == 429
    assert response.json()["error"]["message"] == "rate limited"
```

- [ ] **Step 2: Run route tests**

Run: `python3 -m pytest tests/unit/test_routes_responses.py -q`
Expected: collection failure because `kiro.routes_responses` does not exist.

- [ ] **Step 3: Implement route and error/state orchestration**

```python
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kiro.models_responses import ResponsesRequest
from kiro.responses_adapter import (
    ResponsesConversionError,
    ResponsesStreamAccumulator,
    chat_completion_to_response,
    responses_to_chat_request,
    stream_chat_sse_to_responses,
)
from kiro.routes_openai import chat_completions, verify_api_key

router = APIRouter()

@router.post("/v1/responses", dependencies=[Depends(verify_api_key)])
async def create_response(request: Request, request_data: ResponsesRequest):
    store = request.app.state.responses_store
    history = []
    if request_data.previous_response_id:
        snapshot = store.get(request_data.previous_response_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Unknown or expired previous_response_id")
        if request_data.model != snapshot.model:
            raise HTTPException(status_code=400, detail="model must match previous_response_id")
        history = snapshot.messages

    try:
        chat_request = responses_to_chat_request(request_data, history)
    except ResponsesConversionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chat_result = await chat_completions(request, chat_request)
    response_id = f"resp_{uuid.uuid4().hex}"
    if isinstance(chat_result, StreamingResponse):
        accumulator = ResponsesStreamAccumulator(response_id, chat_request.model)

        async def responses_stream():
            async for frame in stream_chat_sse_to_responses(chat_result.body_iterator, accumulator):
                yield frame
            if accumulator.completed:
                store.put(response_id, chat_request.model, chat_request.messages + [accumulator.assistant_message], accumulator.output)

        return StreamingResponse(responses_stream(), media_type="text/event-stream")

    payload = json.loads(chat_result.body)
    if chat_result.status_code >= 400:
        return JSONResponse(status_code=chat_result.status_code, content=payload)
    response_payload, assistant_message = chat_completion_to_response(payload, response_id)
    store.put(response_id, chat_request.model, chat_request.messages + [assistant_message], response_payload["output"])
    return JSONResponse(content=response_payload)
```

For successful JSON, parse `chat_result.body`, allocate `resp_<uuid>`, call `chat_completion_to_response`, persist `chat_request.messages + [assistant_message]`, and return a JSON response.

For streaming, allocate one response ID and one accumulator. Its wrapper must forward converted event frames; only after natural iteration and `accumulator.completed` may it store `chat_request.messages + [accumulator.assistant_message]`. It must never store failed or disconnected partial output. Preserve `text/event-stream`.

- [ ] **Step 4: Re-run route tests**

Run: `python3 -m pytest tests/unit/test_routes_responses.py -q`
Expected: all new endpoint/auth/JSON/continuation/streaming/error tests pass.

### Task 5: Register state and router without altering existing routes

**Files:**
- Create: `tests/unit/test_main_routes.py`
- Modify: `main.py`

- [ ] **Step 1: Write failing registration tests**

```python
def test_application_registers_responses_router_without_removing_existing_routes():
    from main import app

    route_paths = {route.path for route in app.routes}

    assert "/v1/responses" in route_paths
    assert "/v1/chat/completions" in route_paths
    assert "/v1/messages" in route_paths
    assert "/v1/messages/count_tokens" in route_paths
```

- [ ] **Step 2: Run the registration test**

Run: `python3 -m pytest tests/unit/test_main_routes.py -q`
Expected: failure because `/v1/responses` is not registered.

- [ ] **Step 3: Register the isolated route and per-app state store**

Add imports:

```python
from kiro.responses_adapter import ResponseStateStore
from kiro.routes_responses import router as responses_router
```

At the start of `lifespan`, before yielding, initialize:

```python
app.state.responses_store = ResponseStateStore(max_entries=100, ttl_seconds=7200)
```

Then add exactly one router registration after the existing OpenAI router:

```python
# OpenAI Responses API: /v1/responses
app.include_router(responses_router)
```

Do not reorder, remove, or alter the Anthropic router registration.

- [ ] **Step 4: Re-run registration and Responses route tests**

Run: `python3 -m pytest tests/unit/test_main_routes.py tests/unit/test_routes_responses.py -q`
Expected: all pass.

### Task 6: Document the Codex profile and validate the compatibility boundary

**Files:**
- Modify: `README.md`
- Modify: `tests/unit/test_routes_responses.py`

- [ ] **Step 1: Add a failing OpenAPI/route-boundary test**

```python
def test_responses_endpoint_uses_post_and_existing_anthropic_path_remains_registered(test_client):
    paths = test_client.get("/openapi.json").json()["paths"]
    assert "post" in paths["/v1/responses"]
    assert "post" in paths["/v1/messages"]
    assert "post" in paths["/v1/chat/completions"]
```

- [ ] **Step 2: Run the boundary test**

Run: `python3 -m pytest tests/unit/test_routes_responses.py -q`
Expected: pass after Task 5; this locks in endpoint coexistence before docs are changed.

- [ ] **Step 3: Add README usage instructions without secrets**

Add a “Codex CLI via OpenAI Responses API” section containing:

```toml
# ~/.codex/kiro-gpt.config.toml
model = "<exact gpt-* ID selected by Kiro CLI /model>"
model_provider = "kiro"

[model_providers.kiro]
name = "Kiro Gateway"
base_url = "http://127.0.0.1:8000/v1"
env_key = "KIRO_GATEWAY_API_KEY"
wire_api = "responses"
supports_websockets = false
```

Document:

```bash
export KIRO_GATEWAY_API_KEY='<your gateway proxy key>'
codex --profile kiro-gpt
```

State that `previous_response_id` state is memory-only, capped at 100 responses, expires after two hours, and is cleared on restart. State explicitly that the gateway remains loopback-bound and that this feature does not change Claude Code’s `/v1/messages` configuration.

- [ ] **Step 4: Run focused checks**

Run:

```bash
python3 -m pytest \
  tests/unit/test_models_responses.py \
  tests/unit/test_responses_adapter.py \
  tests/unit/test_routes_responses.py \
  tests/unit/test_main_routes.py -q
git diff --check
```

Expected: all new Responses tests pass and no whitespace errors.

### Task 7: Regression and isolated end-to-end validation

**Files:**
- Modify only if validation identifies a Responses implementation defect.

- [ ] **Step 1: Run all previously healthy OpenAI tests**

Run:

```bash
python3 -m pytest tests/unit/test_routes_openai.py -q
```

Expected: all pass. This verifies the existing Chat execution path that Responses delegates to.

- [ ] **Step 2: Run the full test suite and compare with the documented baseline**

Run:

```bash
set -o pipefail
python3 -m pytest tests -q --tb=no 2>&1 | tee /tmp/kiro-gateway-full-tests.txt
```

Expected: the known 29 Anthropic failures remain the only failures; no Responses test and no previously passing test may fail. The command’s non-zero exit is expected only because the clean baseline already fails.

- [ ] **Step 3: Verify exact failure scope**

Run:

```bash
grep '^FAILED' /tmp/kiro-gateway-full-tests.txt
```

Expected: only the existing `TestExtractThinkingConfigFromAnthropic`, Anthropic route tests sharing the omitted `hidden_models` argument, and the two existing `TestAnthropicHTTPClientSelection` tests appear.

- [ ] **Step 4: Build an isolated loopback-only container (optional but recommended before deployment)**

Run from the worktree:

```bash
docker build -t kiro-gateway:responses-test .
docker run --rm --name kiro-gateway-responses-test \
  -p 127.0.0.1:8001:8000 \
  -v "$HOME/.local/share/kiro-cli:/home/ubuntu/.local/share/kiro-cli:ro" \
  --env-file .env \
  kiro-gateway:responses-test
```

Use a non-secret local test request to confirm `/openapi.json` exposes `/v1/responses`; only send a real Responses request after explicitly verifying the correct exact Kiro CLI model ID. Do not replace the running `kiro-gateway` container at `127.0.0.1:8000` without explicit user approval.

- [ ] **Step 5: Review the final diff and do not commit automatically**

Run:

```bash
git diff --check
git status --short
git diff -- main.py kiro/models_responses.py kiro/responses_adapter.py \
  kiro/routes_responses.py tests/unit README.md
```

Expected: only the scoped Responses implementation, tests, README instructions, and previously approved spec/plan documents are changed. Do not create a commit unless the user explicitly requests one.
