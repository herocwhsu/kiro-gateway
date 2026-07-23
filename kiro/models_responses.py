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


ResponsesTool = Union[ResponsesFunctionTool, Dict[str, Any]]


class ResponsesRequest(BaseModel):
    model: str
    input: ResponsesInput
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    reasoning: Optional[Dict[str, Any]] = None
    tools: Optional[List[ResponsesTool]] = None
    previous_response_id: Optional[str] = None
    stream: bool = False
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    model_config = {"extra": "allow"}

    @property
    def reasoning_effort(self) -> Optional[str]:
        return (self.reasoning or {}).get("effort")
