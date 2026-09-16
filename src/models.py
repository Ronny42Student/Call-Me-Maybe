from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MaskCache:
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
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
