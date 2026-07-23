import json
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from kiro.config import PROXY_API_KEY
from kiro.responses_adapter import ResponseStateStore
from kiro.routes_responses import router


def test_responses_endpoint_converts_chat_completion_and_persists_continuation():
    app = FastAPI()
    app.state.responses_store = ResponseStateStore()
    app.include_router(router)
    chat_response = {
        "model": "gpt-5",
        "choices": [
            {"message": {"role": "assistant", "content": "Hello"}}
        ],
    }

    with patch(
        "kiro.routes_responses.chat_completions",
        new=AsyncMock(return_value=JSONResponse(content=chat_response)),
    ) as chat_completions:
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
                json={"model": "gpt-5", "input": "Hi"},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["output_text"] == "Hello"
    assert app.state.responses_store.get(body["id"]).messages[-1].content == "Hello"
    assert chat_completions.await_args.args[1].model == "gpt-5"


def test_main_registers_responses_endpoint():
    from main import app

    assert any(
        getattr(nested_route, "path", None) == "/v1/responses"
        for included_router in app.routes
        for nested_route in getattr(
            getattr(included_router, "original_router", None), "routes", []
        )
    )


def test_responses_endpoint_rejects_unknown_or_model_mismatched_continuation():
    app = FastAPI()
    app.state.responses_store = ResponseStateStore()
    app.state.responses_store.put("resp_gpt5", "gpt-5", [])
    app.include_router(router)

    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {PROXY_API_KEY}"}
        unknown = client.post(
            "/v1/responses",
            headers=headers,
            json={
                "model": "gpt-5",
                "input": "Continue",
                "previous_response_id": "resp_missing",
            },
        )
        mismatch = client.post(
            "/v1/responses",
            headers=headers,
            json={
                "model": "gpt-4.1",
                "input": "Continue",
                "previous_response_id": "resp_gpt5",
            },
        )

    assert unknown.status_code == 404
    assert mismatch.status_code == 400


def test_responses_endpoint_converts_stream_to_responses_sse():
    async def chat_stream():
        yield b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    app = FastAPI()
    app.state.responses_store = ResponseStateStore()
    app.include_router(router)
    with patch(
        "kiro.routes_responses.chat_completions",
        new=AsyncMock(
            return_value=StreamingResponse(chat_stream(), media_type="text/event-stream")
        ),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
                json={"model": "gpt-5", "input": "Hi", "stream": True},
            )

    assert response.status_code == 200
    assert "event: response.created" in response.text
    assert '"type": "response.output_text.delta"' in response.text
    assert len(app.state.responses_store._snapshots) == 1


def test_responses_endpoint_coexists_with_anthropic_and_chat_completion_routes():
    from main import app

    paths = app.openapi()["paths"]
    assert "post" in paths["/v1/responses"]
    assert "post" in paths["/v1/messages"]
    assert "post" in paths["/v1/chat/completions"]


def test_responses_endpoint_preserves_streaming_chat_error_response():
    app = FastAPI()
    app.state.responses_store = ResponseStateStore()
    app.include_router(router)
    with patch(
        "kiro.routes_responses.chat_completions",
        new=AsyncMock(
            return_value=JSONResponse(
                status_code=400,
                content={"error": {"message": "Invalid model"}},
            )
        ),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
                json={"model": "auto-kiro", "input": "Hi", "stream": True},
            )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Invalid model"


def test_responses_endpoint_streams_codex_custom_exec():
    async def chat_stream():
        yield (
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            b'"id":"call_exec","function":{"name":"exec",'
            b'"arguments":"{\\"input\\":\\"printf TOOL_PROBE_OK\\"}"}}]}}]}\n\n'
        )
        yield b"data: [DONE]\n\n"

    app = FastAPI()
    app.state.responses_store = ResponseStateStore()
    app.include_router(router)
    with patch(
        "kiro.routes_responses.chat_completions",
        new=AsyncMock(
            return_value=StreamingResponse(chat_stream(), media_type="text/event-stream")
        ),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
                json={
                    "model": "gpt-5",
                    "stream": True,
                    "input": [
                        {
                            "type": "additional_tools",
                            "role": "developer",
                            "tools": [
                                {
                                    "type": "custom",
                                    "name": "exec",
                                    "description": "Run a shell command.",
                                    "format": {"type": "grammar", "syntax": "command"},
                                }
                            ],
                        },
                        {"role": "user", "content": "Create a file."},
                    ],
                },
            )

    assert response.status_code == 200
    assert "event: response.custom_tool_call_input.delta" in response.text
    assert '"type": "custom_tool_call"' in response.text
