import copy
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from kiro.models_openai import (
    ChatCompletionRequest,
    ChatMessage,
    Tool,
    ToolFunction,
)
from kiro.models_responses import ResponsesFunctionTool, ResponsesRequest


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

    def _purge_expired(self) -> None:
        cutoff = time.time() - self._ttl_seconds
        expired_ids = [
            response_id
            for response_id, snapshot in self._snapshots.items()
            if snapshot.created_at < cutoff
        ]
        for response_id in expired_ids:
            del self._snapshots[response_id]

    def put(
        self,
        response_id: str,
        model: str,
        messages: List[ChatMessage],
        output: Optional[List[Dict[str, Any]]] = None,
    ) -> ResponseSnapshot:
        self._purge_expired()
        snapshot = ResponseSnapshot(
            response_id=response_id,
            model=model,
            messages=copy.deepcopy(messages),
            output=copy.deepcopy(output or []),
            created_at=time.time(),
        )
        self._snapshots.pop(response_id, None)
        self._snapshots[response_id] = snapshot
        while len(self._snapshots) > self._max_entries:
            self._snapshots.popitem(last=False)
        return copy.deepcopy(snapshot)

    def get(self, response_id: str) -> Optional[ResponseSnapshot]:
        self._purge_expired()
        snapshot = self._snapshots.get(response_id)
        return copy.deepcopy(snapshot) if snapshot else None


class ResponsesConversionError(ValueError):
    pass


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {
                "input_text",
                "output_text",
            }:
                raise ResponsesConversionError("Unsupported message content part")
            texts.append(part.get("text", ""))
        return "".join(texts)

    raise ResponsesConversionError("Message content must be text")


def _messages_from_input(input_data: Any) -> List[ChatMessage]:
    if isinstance(input_data, str):
        return [ChatMessage(role="user", content=input_data)]

    messages = []
    for item in input_data:
        item_type = item.get("type")
        if item_type == "additional_tools":
            continue
        if item_type in {"function_call", "custom_tool_call"}:
            arguments = item.get("arguments", "{}")
            if item_type == "custom_tool_call":
                arguments = json.dumps({"input": item.get("input", "")})
            messages.append(
                ChatMessage(
                    role="assistant",
                    tool_calls=[
                        {
                            "id": item.get("call_id", ""),
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": arguments,
                            },
                        }
                    ],
                )
            )
            continue
        if item_type in {"function_call_output", "custom_tool_call_output"}:
            output = item.get("output", "")
            messages.append(
                ChatMessage(
                    role="tool",
                    tool_call_id=item.get("call_id", ""),
                    content=output if isinstance(output, str) else json.dumps(output),
                )
            )
            continue
        if "role" not in item:
            raise ResponsesConversionError(f"Unsupported input item: {item_type}")
        role = item["role"]
        if role not in {"developer", "system", "user", "assistant"}:
            raise ResponsesConversionError(f"Unsupported message role: {role}")
        messages.append(
            ChatMessage(
                role="system" if role == "developer" else role,
                content=_text_from_content(item.get("content", "")),
            )
        )
    return messages


def _codex_additional_tools(input_data: Any) -> List[Dict[str, Any]]:
    if not isinstance(input_data, list):
        return []
    return [
        tool
        for item in input_data
        if item.get("type") == "additional_tools"
        for tool in item.get("tools", [])
        if isinstance(tool, dict)
    ]


def _flatten_tool_namespaces(declarations: List[Any]) -> List[Any]:
    """Expand ``{"type": "namespace"}`` containers into their nested tools.

    Codex 0.149.1 stopped sending a flat ``tools`` array. It sends no top-level
    ``tools`` key at all; declarations arrive as an ``additional_tools`` input
    item holding ``namespace`` containers (``functions``, ``collaboration``).
    Without this expansion every nested tool hits the ``else: continue`` in
    ``_chat_tools_from_responses`` and the model is handed an empty tool list --
    which it reports as "no terminal tool is available".

    Nested tools keep their BARE name (``exec``, not ``functions.exec``). Two
    reasons, both load-bearing:

    * The Kiro backend rejects a dot in a tool name -- a declaration named
      ``functions.exec`` comes back as HTTP 400 ``Invalid tool use format
      (REQUEST_BODY_INVALID)``, while ``shell``, ``functions__exec`` and
      ``functions-exec`` are all accepted. Verified by probing the endpoint.
    * A bare name is what the client declared the tool as, so the name needs no
      translation on the way back out, nor when the client echoes the call in a
      later request's ``input``. Any rewrite would have to be undone in three
      places to keep a round-trip consistent.

    A namespace prefix is applied only to break a collision, using ``__``
    (dot-free, so it survives the backend's validation).
    """
    flat: List[Any] = []
    seen: set[str] = set()

    def walk(items: List[Any], prefix: str) -> None:
        for declaration in items:
            if not (isinstance(declaration, dict) and declaration.get("type") == "namespace"):
                flat.append(declaration)
                if isinstance(declaration, dict) and isinstance(declaration.get("name"), str):
                    seen.add(declaration["name"])
                continue
            ns = declaration.get("name") or ""
            nested_prefix = f"{prefix}__{ns}" if prefix and ns else (ns or prefix)
            children: List[Any] = []
            for nested in declaration.get("tools") or []:
                if not isinstance(nested, dict):
                    continue
                child = dict(nested)
                name = child.get("name") or ""
                if name and name in seen and nested_prefix:
                    child["name"] = f"{nested_prefix}__{name}"
                children.append(child)
            walk(children, nested_prefix)

    walk(declarations, "")
    return flat


def _responses_tool_declarations(request: ResponsesRequest) -> List[Any]:
    declarations: List[Any] = [*(request.tools or [])]
    declarations.extend(_codex_additional_tools(request.input))
    return _flatten_tool_namespaces(declarations)


def codex_custom_tool_names(request: ResponsesRequest) -> set[str]:
    return {
        declaration["name"]
        for declaration in _responses_tool_declarations(request)
        if isinstance(declaration, dict)
        and declaration.get("type") == "custom"
        and isinstance(declaration.get("name"), str)
    }


def _chat_tools_from_responses(request: ResponsesRequest) -> List[Tool]:
    declarations = _responses_tool_declarations(request)
    tools: List[Tool] = []
    for declaration in declarations:
        if isinstance(declaration, ResponsesFunctionTool):
            name = declaration.name
            description = declaration.description
            parameters = declaration.parameters
        elif isinstance(declaration, dict) and declaration.get("type") == "function":
            name = declaration.get("name", "")
            description = declaration.get("description")
            parameters = declaration.get("parameters")
        elif isinstance(declaration, dict) and declaration.get("type") == "custom":
            name = declaration.get("name", "")
            description = declaration.get("description")
            parameters = {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": False,
            }
        else:
            continue
        if name:
            tools.append(
                Tool(
                    type="function",
                    function=ToolFunction(
                        name=name,
                        description=description,
                        parameters=parameters,
                    ),
                )
            )
    return tools


def responses_to_chat_request(request: ResponsesRequest) -> ChatCompletionRequest:
    messages = []
    if request.instructions:
        messages.append(ChatMessage(role="system", content=request.instructions))
    messages.extend(_messages_from_input(request.input))

    tools = _chat_tools_from_responses(request) or None

    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        stream=request.stream,
        temperature=request.temperature,
        top_p=request.top_p,
        max_completion_tokens=request.max_output_tokens,
        reasoning_effort=request.reasoning_effort,
        tools=tools,
        tool_choice=request.tool_choice,
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _custom_tool_input(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
        if isinstance(parsed, dict) and isinstance(parsed.get("input"), str):
            return parsed["input"]
        return arguments
    return json.dumps(arguments)


def chat_completion_to_response(
    chat_response: Any,
    response_id: str,
    custom_tool_names: Optional[set[str]] = None,
) -> tuple[Dict[str, Any], ChatMessage]:
    custom_tool_names = custom_tool_names or set()
    choices = _field(chat_response, "choices", [])
    choice = choices[0] if choices else {}
    message_data = _field(choice, "message", {})
    content = _field(message_data, "content") or ""
    tool_calls = _field(message_data, "tool_calls") or []
    assistant_message = ChatMessage(
        role="assistant",
        content=content,
        tool_calls=copy.deepcopy(tool_calls) or None,
    )

    output: List[Dict[str, Any]] = []
    if content:
        output.append(
            {
                "id": f"msg_{response_id}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        )
    for tool_call in tool_calls:
        function = _field(tool_call, "function", {})
        name = _field(function, "name", "")
        arguments = _field(function, "arguments", "{}")
        if name in custom_tool_names:
            output.append(
                {
                    "id": _field(tool_call, "id", ""),
                    "type": "custom_tool_call",
                    "status": "completed",
                    "call_id": _field(tool_call, "id", ""),
                    "name": name,
                    "input": _custom_tool_input(arguments),
                }
            )
            continue
        output.append(
            {
                "id": _field(tool_call, "id", ""),
                "type": "function_call",
                "status": "completed",
                "call_id": _field(tool_call, "id", ""),
                "name": name,
                "arguments": arguments,
            }
        )

    usage = _field(chat_response, "usage")
    response: Dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "status": "completed",
        "model": _field(chat_response, "model", ""),
        "output": output,
        "output_text": content,
    }
    if usage is not None:
        response["usage"] = {
            "input_tokens": _field(usage, "prompt_tokens", 0),
            "output_tokens": _field(usage, "completion_tokens", 0),
            "total_tokens": _field(usage, "total_tokens", 0),
        }
    return response, assistant_message


def _sse_event(event_type: str, payload: Dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps({'type': event_type, **payload})}\n\n"


async def chat_stream_to_responses(
    chat_stream: AsyncIterator[bytes | str],
    response_id: str,
    model: str,
    on_complete: Optional[Callable[[Dict[str, Any], ChatMessage], Awaitable[None]]] = None,
    custom_tool_names: Optional[set[str]] = None,
) -> AsyncIterator[str]:
    custom_tool_names = custom_tool_names or set()
    response = {
        "id": response_id,
        "object": "response",
        "status": "in_progress",
        "model": model,
        "output": [],
        "output_text": "",
    }
    yield _sse_event("response.created", {"response": response})
    yield _sse_event("response.in_progress", {"response": response})

    message_item: Optional[Dict[str, Any]] = None
    tool_items: Dict[int, Dict[str, Any]] = {}
    # Keyed by the upstream `index`, not append-ordered. Every read below is
    # `assistant_tool_calls[index]`, so a list only worked while indices arrived
    # dense, zero-based and in order: an upstream that opens with index 1 would
    # append at position 0 and then IndexError on [1], which the broad except
    # turns into response.failed. tool_items above was already a dict for the
    # same reason; this just stops the two from disagreeing.
    assistant_tool_calls: Dict[int, Dict[str, Any]] = {}
    output_text = ""
    try:
        async for raw_chunk in chat_stream:
            chunk = raw_chunk.decode() if isinstance(raw_chunk, bytes) else raw_chunk
            for line in chunk.splitlines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    continue
                chat_event = json.loads(data)
                choices = chat_event.get("choices", [])
                delta = choices[0].get("delta", {}) if choices else {}
                for tool_delta in delta.get("tool_calls", []):
                    index = tool_delta.get("index", 0)
                    function = tool_delta.get("function", {})
                    item = tool_items.get(index)
                    if item is None:
                        call_id = tool_delta.get("id", f"call_{response_id}_{index}")
                        name = function.get("name", "")
                        is_custom_tool = name in custom_tool_names
                        item = {
                            "id": f"fc_{call_id}",
                            "type": "custom_tool_call" if is_custom_tool else "function_call",
                            "status": "in_progress",
                            "call_id": call_id,
                            "name": name,
                            "input" if is_custom_tool else "arguments": "",
                        }
                        tool_items[index] = item
                        assistant_tool_calls[index] = {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item["name"],
                                "arguments": "",
                            },
                        }
                        response["output"].append(item)
                        yield _sse_event(
                            "response.output_item.added",
                            {"output_index": len(response["output"]) - 1, "item": item},
                        )
                    if function.get("name"):
                        item["name"] = function["name"]
                        assistant_tool_calls[index]["function"]["name"] = function["name"]
                    arguments_delta = function.get("arguments", "")
                    if arguments_delta:
                        assistant_tool_calls[index]["function"]["arguments"] += arguments_delta
                        if item["type"] == "custom_tool_call":
                            continue
                        item["arguments"] += arguments_delta
                        yield _sse_event(
                            "response.function_call_arguments.delta",
                            {
                                "item_id": item["id"],
                                "output_index": response["output"].index(item),
                                "delta": arguments_delta,
                            },
                        )
                text = delta.get("content")
                if not text:
                    continue
                if message_item is None:
                    message_item = {
                        "id": f"msg_{response_id}",
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                            }
                        ],
                    }
                    response["output"].append(message_item)
                    yield _sse_event(
                        "response.output_item.added",
                        {"output_index": 0, "item": message_item},
                    )
                    yield _sse_event(
                        "response.content_part.added",
                        {
                            "item_id": message_item["id"],
                            "output_index": 0,
                            "content_index": 0,
                            "part": message_item["content"][0],
                        },
                    )
                output_text += text
                message_item["content"][0]["text"] = output_text
                yield _sse_event(
                    "response.output_text.delta",
                    {
                        "item_id": message_item["id"],
                        "output_index": 0,
                        "content_index": 0,
                        "delta": text,
                    },
                )
    except Exception as error:
        response["status"] = "failed"
        yield _sse_event("response.failed", {"response": response, "error": str(error)})
        return

    if message_item is not None:
        message_item["status"] = "completed"
        yield _sse_event(
            "response.content_part.done",
            {
                "item_id": message_item["id"],
                "output_index": 0,
                "content_index": 0,
                "part": message_item["content"][0],
            },
        )
        yield _sse_event(
            "response.output_item.done",
            {"output_index": 0, "item": message_item},
        )
    for index, item in tool_items.items():
        item["status"] = "completed"
        output_index = response["output"].index(item)
        if item["type"] == "custom_tool_call":
            tool_input = _custom_tool_input(
                assistant_tool_calls[index]["function"]["arguments"]
            )
            item["input"] = tool_input
            yield _sse_event(
                "response.custom_tool_call_input.delta",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "delta": tool_input,
                },
            )
        else:
            yield _sse_event(
                "response.function_call_arguments.done",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "arguments": item["arguments"],
                },
            )
        yield _sse_event(
            "response.output_item.done",
            {"output_index": output_index, "item": item},
        )
    response["status"] = "completed"
    response["output_text"] = output_text
    # Sort by index so the replayed order matches what upstream sent, not the
    # order deltas happened to arrive in.
    ordered_tool_calls = [assistant_tool_calls[i] for i in sorted(assistant_tool_calls)]
    assistant_message = ChatMessage(
        role="assistant",
        content=output_text,
        tool_calls=ordered_tool_calls or None,
    )
    if on_complete is not None:
        await on_complete(response, assistant_message)
    yield _sse_event("response.completed", {"response": response})
