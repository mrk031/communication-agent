from typing import List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class HardWareParameterSchema(BaseModel):
    """Information about a parameter."""

    # ^ Doc-string for the entity HardWareParameterSchema.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(..., description="The name of the parameter")
    value: Optional[str] = Field(
        ..., description="The value of the parameter"
    )
    description: Optional[str] = Field(..., description="description of the parameter")


