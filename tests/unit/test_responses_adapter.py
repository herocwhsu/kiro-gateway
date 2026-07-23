import json

import pytest

from kiro.models_responses import ResponsesRequest
from kiro.responses_adapter import (
    ResponseStateStore,
    ResponsesConversionError,
    chat_completion_to_response,
    chat_stream_to_responses,
    responses_to_chat_request,
)


def test_converts_string_input_to_chat_request_with_exact_model_id():
    request = ResponsesRequest.model_validate(
        {
            "model": "gpt-5",
            "input": "Hello",
        }
    )

    chat_request = responses_to_chat_request(request)

    assert chat_request.model == "gpt-5"
    assert [(message.role, message.content) for message in chat_request.messages] == [
        ("user", "Hello")
    ]


def test_converts_instructions_and_message_items_to_chat_messages():
    request = ResponsesRequest.model_validate(
        {
            "model": "gpt-5",
            "instructions": "System rules",
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Developer rules"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Question"}],
                },
            ],
        }
    )

    chat_request = responses_to_chat_request(request)

    assert [(message.role, message.content) for message in chat_request.messages] == [
        ("system", "System rules"),
        ("system", "Developer rules"),
        ("user", "Question"),
    ]


def test_converts_function_call_and_function_call_output_to_chat_tool_messages():
    request = ResponsesRequest.model_validate(
        {
            "model": "gpt-5",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": "{\"path\":\"a.py\"}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": {"contents": "text"},
                },
            ],
        }
    )

    chat_request = responses_to_chat_request(request)

    assert chat_request.messages[0].tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{\"path\":\"a.py\"}"},
        }
    ]
    assert chat_request.messages[1].role == "tool"
    assert chat_request.messages[1].tool_call_id == "call_1"
    assert chat_request.messages[1].content == '{"contents": "text"}'


def test_converts_function_tools_and_generation_controls_to_chat_request():
    request = ResponsesRequest.model_validate(
        {
            "model": "gpt-5",
            "input": "Inspect the file",
            "stream": True,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 123,
            "reasoning": {"effort": "medium"},
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object"},
                }
            ],
        }
    )

    chat_request = responses_to_chat_request(request)

    assert chat_request.stream is True
    assert chat_request.temperature == 0.2
    assert chat_request.top_p == 0.9
    assert chat_request.max_completion_tokens == 123
    assert chat_request.reasoning_effort == "medium"
    assert chat_request.tool_choice == "required"
    assert chat_request.tools[0].function.name == "read_file"




def test_converts_codex_additional_tools_custom_exec_to_chat_tool():
    request = ResponsesRequest.model_validate(
        {
            "model": "gpt-5",
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
        }
    )

    chat_request = responses_to_chat_request(request)

    assert [(message.role, message.content) for message in chat_request.messages] == [
        ("user", "Create a file.")
    ]
    assert chat_request.tools[0].function.name == "exec"
    assert chat_request.tools[0].function.parameters == {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
        "additionalProperties": False,
    }
def test_response_state_store_evicts_oldest_entry_at_capacity():
    store = ResponseStateStore(max_entries=1, ttl_seconds=7200)

    store.put("resp_old", "gpt-5", [])
    store.put("resp_new", "gpt-5", [])

    assert store.get("resp_old") is None
    assert store.get("resp_new").model == "gpt-5"


def test_response_state_store_expires_entries(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("kiro.responses_adapter.time.time", lambda: now[0])
    store = ResponseStateStore(max_entries=100, ttl_seconds=10)

    store.put("resp_expired", "gpt-5", [])
    now[0] = 11.0

    assert store.get("resp_expired") is None


def test_converts_chat_completion_to_responses_text_and_function_call_items():
    response, assistant_message = chat_completion_to_response(
        {
            "model": "gpt-5",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I will inspect it.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": "{\"path\":\"a.py\"}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        },
        "resp_test",
    )

    assert response["id"] == "resp_test"
    assert response["object"] == "response"
    assert response["status"] == "completed"
    assert response["output_text"] == "I will inspect it."
    assert [item["type"] for item in response["output"]] == [
        "message",
        "function_call",
    ]
    assert response["output"][1]["call_id"] == "call_1"
    assert assistant_message.tool_calls[0]["function"]["name"] == "read_file"


def test_rejects_unsupported_input_item_with_its_type():
    request = ResponsesRequest.model_validate(
        {"model": "gpt-5", "input": [{"type": "computer_call"}]}
    )

    with pytest.raises(ResponsesConversionError, match="computer_call"):
        responses_to_chat_request(request)


@pytest.mark.asyncio
async def test_converts_chat_sse_to_responses_sse_events():
    async def chat_stream():
        yield 'data: {"model":"gpt-5","choices":[{"delta":{"content":"Hello"}}]}\n\n'
        yield "data: [DONE]\n\n"

    events = [
        json.loads(line.removeprefix("data: "))
        async for chunk in chat_stream_to_responses(chat_stream(), "resp_stream", "gpt-5")
        for line in chunk.splitlines()
        if line.startswith("data: ")
    ]

    assert [event["type"] for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[4]["delta"] == "Hello"
    assert events[-1]["response"]["output_text"] == "Hello"


@pytest.mark.asyncio
async def test_converts_chat_tool_call_deltas_to_responses_function_call_events():
    async def chat_stream():
        yield (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"id":"call_1","function":{"name":"read_file",'
            '"arguments":"{\\\"path\\\":\\\"a.py\\\"}"}}]}}]}\n\n'
        )

    events = [
        json.loads(line.removeprefix("data: "))
        async for chunk in chat_stream_to_responses(chat_stream(), "resp_stream", "gpt-5")
        for line in chunk.splitlines()
        if line.startswith("data: ")
    ]

    assert "response.function_call_arguments.delta" in [
        event["type"] for event in events
    ]
    assert events[-1]["response"]["output"][0]["type"] == "function_call"
    assert events[-1]["response"]["output"][0]["arguments"] == '{"path":"a.py"}'


def test_ignores_codex_builtin_tools_when_converting_to_chat_request():
    request = ResponsesRequest.model_validate(
        {
            "model": "gpt-5",
            "input": "Hello",
            "tools": [
                {"type": "function", "name": "shell", "parameters": {}},
                {"type": "namespace", "name": "functions"},
                {"type": "web_search", "external_web_access": False},
            ],
        }
    )

    chat_request = responses_to_chat_request(request)

    assert [tool.function.name for tool in chat_request.tools] == ["shell"]


def test_serializes_codex_custom_exec_as_custom_tool_call():
    response, assistant_message = chat_completion_to_response(
        {
            "model": "gpt-5",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_exec",
                                "type": "function",
                                "function": {
                                    "name": "exec",
                                    "arguments": '{"input":"printf TOOL_PROBE_OK"}',
                                },
                            }
                        ],
                    }
                }
            ],
        },
        "resp_test",
        custom_tool_names={"exec"},
    )

    assert response["output"] == [
        {
            "id": "call_exec",
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": "call_exec",
            "name": "exec",
            "input": "printf TOOL_PROBE_OK",
        }
    ]
    assert assistant_message.tool_calls[0]["function"]["name"] == "exec"


@pytest.mark.asyncio
async def test_streams_codex_custom_exec_as_custom_tool_call():
    async def chat_stream():
        yield (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"id":"call_exec","function":{"name":"exec",'
            '"arguments":"{\\"input\\":\\"printf TOOL_PROBE_OK\\"}"}}]}}]}\n\n'
        )
        yield "data: [DONE]\n\n"

    events = [
        json.loads(line.removeprefix("data: "))
        async for chunk in chat_stream_to_responses(
            chat_stream(), "resp_stream", "gpt-5", custom_tool_names={"exec"}
        )
        for line in chunk.splitlines()
        if line.startswith("data: ")
    ]

    assert "response.custom_tool_call_input.delta" in [
        event["type"] for event in events
    ]
    assert events[-1]["response"]["output"] == [
        {
            "id": "fc_call_exec",
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": "call_exec",
            "name": "exec",
            "input": "printf TOOL_PROBE_OK",
        }
    ]


def test_converts_codex_custom_tool_call_and_output_to_chat_messages():
    request = ResponsesRequest.model_validate(
        {
            "model": "gpt-5",
            "input": [
                {
                    "type": "custom_tool_call",
                    "call_id": "call_exec",
                    "name": "exec",
                    "input": "cat tool-probe.txt",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_exec",
                    "output": "TOOL_PROBE_OK",
                },
            ],
        }
    )

    chat_request = responses_to_chat_request(request)

    assert chat_request.messages[0].tool_calls == [
        {
            "id": "call_exec",
            "type": "function",
            "function": {"name": "exec", "arguments": '{"input": "cat tool-probe.txt"}'},
        }
    ]
    assert chat_request.messages[1].role == "tool"
    assert chat_request.messages[1].tool_call_id == "call_exec"
    assert chat_request.messages[1].content == "TOOL_PROBE_OK"


def test_converts_codex_additional_function_tools_to_chat_tools():
    request = ResponsesRequest.model_validate(
        {
            "model": "gpt-5",
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [
                        {
                            "type": "function",
                            "name": "wait",
                            "description": "Wait for an operation.",
                            "parameters": {"type": "object", "properties": {}},
                            "strict": True,
                        }
                    ],
                },
                {"role": "user", "content": "Wait."},
            ],
        }
    )

    chat_request = responses_to_chat_request(request)

    assert chat_request.tools[0].function.name == "wait"
    assert chat_request.tools[0].function.parameters == {
        "type": "object",
        "properties": {},
    }
