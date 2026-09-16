"""Helpers for resolving the function name during generation."""

import string
from typing import Any

from src.parser.prompts import build_function_scoped_prompt


def get_allowed_chars(current_str: str, allowed_names: list[str]) -> list[str]:
    """Compute the allowed continuations while generating the name field.

    Args:
        current_str: The JSON generated so far.
        allowed_names: Names of the functions the model may call.

    Returns:
        List of literal string continuations still valid at this point
        (e.g. remaining prefix characters, or the printable alphabet
        once the name field is fully resolved).
    """
    prefix = '{"name":"'
    if len(current_str) < len(prefix):
        return [prefix[len(current_str):]]

    after_prefix = current_str[len(prefix):]
    if '"' not in after_prefix:
        return [name[len(after_prefix):] + '"'
                for name in allowed_names if name.startswith(after_prefix)]

    func_name = after_prefix.split('"')[0]
    target = prefix + func_name + '","parameters":{'
    if len(current_str) < len(target):
        return [target[len(current_str):]]

    return list(string.printable)


def resolve_function_name(
        current_str: str, allowed_fn: list[str],
        prefix: str) -> str | None:
    """Force-complete the function name once it is unambiguous.

    Args:
        current_str: The JSON generated so far.
        allowed_fn: Names of the functions the model may call.
        prefix: The literal JSON prefix preceding the name value.

    Returns:
        The remaining characters to append to complete the function
        name (plus closing quote), or None if the name is not yet
        unambiguous or already resolved.
    """
    if prefix not in current_str or '","parameters":{' in current_str:
        return None

    after_prefix = current_str.split(prefix)[1]
    possible_names = [
        n for n in allowed_fn if n.startswith(after_prefix)]

    if len(possible_names) == 1 and possible_names[0] != after_prefix:
        return possible_names[0][len(after_prefix):] + '"'

    return None


def inject_parameters_bridge(
        current_str: str, input_ids: list[int],
        cache: Any, prompt_text: str) -> tuple[str, list[int], bool]:
    """Transition generation from the name field into the parameters object.

    Appends the JSON bridge into "parameters", and either closes the
    call immediately (if the function takes no parameters) or
    re-encodes a function-scoped prompt to continue generation.

    Args:
        current_str: The JSON generated so far.
        input_ids: Token ids encoded so far.
        cache: Object exposing ``func_params``, ``raw_functions`` and
            ``model``.
        prompt_text: The user's natural-language request.

    Returns:
        A tuple of (updated JSON string, updated token ids, whether
        generation is already finished).
    """
    bridge = ',"parameters":{'
    current_str += bridge

    func_name = current_str.split('"name":"')[1].split('"')[0]
    if cache.func_params.get(func_name, 99) == 0:
        current_str += "}}"
        return current_str, input_ids, True

    active_schema = next(
        (f for f in cache.raw_functions if f["name"] == func_name),
        None
    )

    if active_schema:
        tiny_prompt = build_function_scoped_prompt(
            prompt_text, current_str, active_schema)
        input_ids = cache.model.encode(tiny_prompt).tolist()[0]
    else:
        input_ids.extend(cache.model.encode(bridge).tolist()[0])

    return current_str, input_ids, False
