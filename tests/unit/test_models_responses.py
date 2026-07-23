import pytest
from pydantic import ValidationError

from kiro.models_responses import ResponsesRequest


def test_accepts_codex_text_request_and_unknown_metadata():
    request = ResponsesRequest.model_validate(
        {
            "model": "gpt-5",
            "instructions": "Be concise.",
            "input": "Hello",
            "max_output_tokens": 200,
            "reasoning": {"effort": "high"},
            "metadata": {"codex": "preserved"},
        }
    )

    assert request.model == "gpt-5"
    assert request.input == "Hello"
    assert request.reasoning_effort == "high"
    assert request.model_extra["metadata"] == {"codex": "preserved"}


def test_accepts_message_items_and_function_tools():
    request = ResponsesRequest.model_validate(
        {
            "model": "gpt-5",
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Rules"}],
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "done"},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "parameters": {"type": "object"},
                }
            ],
        }
    )

    assert request.input[0]["role"] == "developer"
    assert request.tools[0].name == "read_file"


def test_accepts_streaming_continuation_and_generation_controls():
    request = ResponsesRequest.model_validate(
        {
            "model": "gpt-5",
            "input": "Continue",
            "previous_response_id": "resp_previous",
            "stream": "true",
            "tool_choice": "required",
            "temperature": "0.2",
            "top_p": "0.9",
        }
    )

    assert request.previous_response_id == "resp_previous"
    assert request.stream is True
    assert request.tool_choice == "required"
    assert request.temperature == 0.2
    assert request.top_p == 0.9


def test_accepts_codex_builtin_tool_declarations():
    request = ResponsesRequest.model_validate(
        {
            "model": "gpt-5",
            "input": "Hello",
            "tools": [
                {"type": "function", "name": "shell", "parameters": {}},
                {
                    "type": "namespace",
                    "name": "functions",
                    "description": "Codex tool namespace",
                },
                {"type": "web_search", "external_web_access": False},
            ],
        }
    )

    assert request.tools[0].name == "shell"
    assert request.tools[1]["type"] == "namespace"
    assert request.tools[2]["type"] == "web_search"
