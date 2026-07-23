import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kiro.models_openai import ChatCompletionRequest
from kiro.models_responses import ResponsesRequest
from kiro.responses_adapter import (
    ResponseStateStore,
    chat_completion_to_response,
    chat_stream_to_responses,
    codex_custom_tool_names,
    responses_to_chat_request,
)
from kiro.routes_openai import chat_completions, verify_api_key

router = APIRouter()


def _response_body(result: Any) -> dict:
    if isinstance(result, JSONResponse):
        return json.loads(result.body)
    if isinstance(result, dict):
        return result
    raise HTTPException(status_code=502, detail="Unexpected Chat Completions response")


@router.post("/v1/responses", dependencies=[Depends(verify_api_key)])
async def create_response(request: Request, request_data: ResponsesRequest):
    store: ResponseStateStore = request.app.state.responses_store
    chat_request = responses_to_chat_request(request_data)
    custom_tool_names = codex_custom_tool_names(request_data)

    if request_data.previous_response_id:
        snapshot = store.get(request_data.previous_response_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Unknown or expired previous_response_id")
        if snapshot.model != request_data.model:
            raise HTTPException(
                status_code=400,
                detail="previous_response_id model does not match request model",
            )
        chat_request = ChatCompletionRequest(
            **chat_request.model_dump(exclude={"messages"}),
            messages=[*snapshot.messages, *chat_request.messages],
        )

    chat_result = await chat_completions(request, chat_request)
    response_id = f"resp_{uuid.uuid4().hex}"
    if request_data.stream:
        if isinstance(chat_result, JSONResponse) and chat_result.status_code >= 400:
            return chat_result
        if not isinstance(chat_result, StreamingResponse):
            raise HTTPException(status_code=502, detail="Expected streaming Chat Completions response")

        async def persist_stream_response(response: dict, assistant_message) -> None:
            store.put(
                response_id,
                request_data.model,
                [*chat_request.messages, assistant_message],
                response["output"],
            )

        return StreamingResponse(
            chat_stream_to_responses(
                chat_result.body_iterator,
                response_id,
                request_data.model,
                persist_stream_response,
                custom_tool_names,
            ),
            media_type="text/event-stream",
        )

    if isinstance(chat_result, JSONResponse) and chat_result.status_code >= 400:
        return chat_result
    response, assistant_message = chat_completion_to_response(
        _response_body(chat_result), response_id, custom_tool_names
    )
    store.put(
        response_id,
        request_data.model,
        [*chat_request.messages, assistant_message],
        response["output"],
    )
    return JSONResponse(content=response)
