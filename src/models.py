"""Pydantic models shared across the function calling pipeline."""

from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MaskCache:
    """Precomputed data shared across the constrained decoding process.

    Attributes:
        model: LLM wrapper used for encoding and inference.
        vocab_dict: Mapping from token id to token string.
        allowed_fn: Names of the functions the model may call.
        raw_functions: Raw function definitions as loaded from JSON.
        func_params: Mapping from function name to its parameter count.
        param_types: Mapping from function name to a dict of
            parameter name -> declared type.
        p4_mask: Boolean mask over the vocabulary for printable tokens.
        p4_numbers_only: Boolean mask restricted to numeric tokens.
        p4_no_comma: Same as ``p4_mask`` but excluding tokens with a comma.
        mini_dict: Reduced (id, token) pairs used for name resolution.
        clean_dict_items: All (id, token) pairs kept after filtering.
    """

    model: Any
    vocab_dict: Dict[int, str]

    allowed_fn: List[str]
    raw_functions: List[Dict[str, Any]]
    func_params: Dict[str, int]
    param_types: Dict[str, Dict[str, Any]]

    p4_mask: np.ndarray[Any, Any]
    p4_numbers_only: np.ndarray[Any, Any]
    p4_no_comma: np.ndarray[Any, Any]
    mini_dict: List[Tuple[int, str]]
    clean_dict_items: List[Tuple[int, str]]


class FunctionDef(BaseModel):
    """Schema of a single callable function definition.

    Attributes:
        name: Function name.
        description: Human-readable description of the function.
        parameters: Mapping of parameter name to its type schema.
        returns: Type schema of the function's return value.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]


class FunctionCallResult(BaseModel):
    """Validated result of resolving a prompt to a function call.

    Attributes:
        prompt: The original natural-language prompt.
        name: The name of the resolved function.
        parameters: The extracted and coerced call arguments.
    """

    prompt: str
    name: str
    parameters: Dict[str, Any]
