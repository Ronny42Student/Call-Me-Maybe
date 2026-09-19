"""Build vocabulary masks and function metadata for constrained decoding."""

import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import ValidationError

from src.models import FunctionDef, MaskCache
from src.vocab import (
    build_vocab_dict,
    compute_vocab_size,
    filter_printable_tokens,
)


def extract_function_metadata(
    raw_functions: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, int], Dict[str, Dict[str, Any]]]:
    """Validate function definitions and extract generation metadata.

    Each raw definition is validated against ``FunctionDef``. On
    failure, an error is printed to stderr and the program exits.

    Args:
        raw_functions: List of raw function definitions.

    Returns:
        A tuple of (allowed function names, parameter counts per
        function, parameter types per function).
    """
    allowed_fn_names: List[str] = []
    func_params: Dict[str, int] = {}
    param_types: Dict[str, Dict[str, Any]] = {}

    for fn in raw_functions:
        try:
            func = FunctionDef(**fn)
            allowed_fn_names.append(func.name)
            func_params[func.name] = len(func.parameters)

            params = fn.get("parameters", {})
            param_types[func.name] = {
                param_key: details.get("type")
                for param_key, details in params.items()
            }

        except ValidationError as e:
            print(
                "Validation failed: input data is invalid or incomplete.",
                file=sys.stderr,
            )
            print(e.errors()[0])
            sys.exit(0)

        except Exception as e:
            print(
                f"An unexpected error occured.\nDetails: {e}",
                file=sys.stderr,
            )
            sys.exit(0)

    return allowed_fn_names, func_params, param_types


def build_number_mask(
    vocab_size: int, clean_dict_items: List[Tuple[int, str]]
) -> np.ndarray[Any, Any]:
    """Build a boolean mask selecting tokens usable inside a number value.

    Args:
        vocab_size: Total number of tokens in the vocabulary.
        clean_dict_items: (id, token) pairs to evaluate.

    Returns:
        Boolean array of shape ``(vocab_size,)``, True for tokens made
        only of digits/numeric punctuation, or equal to "null".
    """
    p4_numbers_only = np.zeros(vocab_size, dtype=bool)
    allowed_math_chars = set("0123456789.-, }")

    for i, s in clean_dict_items:
        if all(char in allowed_math_chars for char in s) or s == "null":
            p4_numbers_only[i] = True

    return p4_numbers_only


def build_masks(
    vocab_size: int,
    valid_ids: List[int],
    clean_dict_items: List[Tuple[int, str]],
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Build the base vocabulary masks used during constrained decoding.

    Args:
        vocab_size: Total number of tokens in the vocabulary.
        valid_ids: Token ids considered printable/valid.
        clean_dict_items: (id, token) pairs to evaluate.

    Returns:
        A tuple of (mask of all valid tokens, mask of numeric-only
        tokens, mask of valid tokens excluding any containing a comma).
    """
    p4_mask = np.zeros(vocab_size, dtype=bool)
    p4_mask[valid_ids] = True

    p4_numbers_only = build_number_mask(vocab_size, clean_dict_items)

    p4_no_comma = p4_mask.copy()
    for i, s in clean_dict_items:
        if "," in s:
            p4_no_comma[i] = False

    return p4_mask, p4_numbers_only, p4_no_comma


def build_mini_dict(
    allowed_fn_names: List[str], clean_dict_items: List[Tuple[int, str]]
) -> List[Tuple[int, str]]:
    """Restrict tokens to those usable while resolving the function name.

    Args:
        allowed_fn_names: Names of the functions the model may call.
        clean_dict_items: (id, token) pairs to evaluate.

    Returns:
        The subset of (id, token) pairs where the token is a prefix of
        one of the allowed function names or a JSON structural phrase.
    """
    target_phrases = allowed_fn_names + [
        '{"name":"',
        '","parameters":{',
        "}",
    ]

    return [
        (i, s)
        for i, s in clean_dict_items
        if any(s in phrase for phrase in target_phrases)
    ]


def build_mask_cache(
    model: Any, raw_functions: List[Dict[str, Any]]
) -> MaskCache:
    """Assemble all precomputed data needed for constrained decoding.

    Args:
        model: LLM wrapper used for encoding and inference.
        raw_functions: List of raw function definitions.

    Returns:
        A fully populated ``MaskCache``.
    """
    vocab_dict = build_vocab_dict(model)
    valid_ids, clean_dict_items = filter_printable_tokens(vocab_dict)

    allowed_fn_names, func_params, param_types = extract_function_metadata(
        raw_functions
    )

    vocab_size = compute_vocab_size(model)
    p4_mask, p4_numbers_only, p4_no_comma = build_masks(
        vocab_size, valid_ids, clean_dict_items
    )

    mini_dict = build_mini_dict(allowed_fn_names, clean_dict_items)

    return MaskCache(
        model=model,
        vocab_dict=vocab_dict,
        allowed_fn=allowed_fn_names,
        raw_functions=raw_functions,
        func_params=func_params,
        param_types=param_types,
        p4_mask=p4_mask,
        p4_numbers_only=p4_numbers_only,
        p4_no_comma=p4_no_comma,
        mini_dict=mini_dict,
        clean_dict_items=clean_dict_items,
    )
