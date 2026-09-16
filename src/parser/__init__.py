"""Constrained JSON generation loop for function calling."""

from typing import Any
import numpy as np

from src.parser.prompts import (
    build_full_prompt,
    build_function_scoped_prompt,
)
from src.parser.name_resolution import (
    get_allowed_chars,
    resolve_function_name,
    inject_parameters_bridge,
)
from src.parser.mask import compute_generation_mask

__all__ = [
    "build_full_prompt",
    "build_function_scoped_prompt",
    "get_allowed_chars",
    "resolve_function_name",
    "inject_parameters_bridge",
    "compute_generation_mask",
    "generate_constrained_json",
]


def generate_constrained_json(prompt_text: str, cache: Any) -> str:
    """Generate a schema-compliant JSON function call via constrained decoding.

    Drives the token-by-token generation loop: resolves the function
    name, injects the parameters bridge, then masks the model's
    logits at each step so only tokens compatible with valid JSON and
    the target schema can be selected, until the call is complete or
    the token budget is exhausted.

    Args:
        prompt_text: The user's natural-language request.
        cache: Precomputed mask/model data (see ``MaskCache``).

    Returns:
        The generated JSON string representing the function call.
    """
    prompt = build_full_prompt(prompt_text, cache)

    input_ids = cache.model.encode(prompt).tolist()[0]
    vocab_size = len(cache.model.get_logits_from_input_ids(input_ids))

    prefix = '{"name":"'
    current_str = prefix

    input_ids.extend(cache.model.encode(prefix).tolist()[0])
    bridge_injected = False

    max_tokens = 150

    while (not current_str.replace(" ", "").replace("\n", "").endswith('}}')
           and len(input_ids) < len(prompt) + max_tokens):

        remainder = resolve_function_name(
            current_str, cache.allowed_fn, prefix)
        if remainder is not None:
            current_str += remainder
            input_ids.extend(cache.model.encode(remainder).tolist()[0])
            continue

        if (current_str.endswith('"')
            and not bridge_injected
            and prefix in current_str
                and len(current_str) > len(prefix)):

            current_str, input_ids, is_finished = inject_parameters_bridge(
                current_str, input_ids, cache, prompt_text)
            bridge_injected = True

            if is_finished:
                break

            continue

        mask, forced_close = compute_generation_mask(
            current_str, cache, vocab_size)

        if forced_close is not None:
            current_str = forced_close
            print(f"\rGenerating: {current_str}", end="", flush=True)
            break

        logits = np.array(cache.model.get_logits_from_input_ids(input_ids))
        logits[~mask] = -np.inf

        best_id = int(np.argmax(logits))
        current_str += cache.vocab_dict.get(best_id, "")
        input_ids.append(best_id)

        if (current_str.endswith('"')
                and not bridge_injected
                and prefix in current_str):
            bridge = ',"parameters":{'
            current_str += bridge
            input_ids.extend(cache.model.encode(bridge).tolist()[0])
            bridge_injected = True
            continue

        else:
            print(f"\rGenerating: {current_str}", end="", flush=True)

    print()
    return current_str
