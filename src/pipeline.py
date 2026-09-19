"""Prompt processing pipeline: generation, validation and reporting."""

import json
import sys
from datetime import datetime
from typing import Any, Dict, List

from pydantic import ValidationError
from rich import print_json

from src.models import FunctionCallResult, MaskCache
from src.parser import generate_constrained_json


def coerce_parameter_types(
    parameters: Dict[str, Any], expected_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Coerce parameter values to match their expected schema types.

    Integers are converted to floats where the schema expects a
    "number", and strings are stripped of surrounding whitespace.

    Args:
        parameters: Raw parameter values extracted from the model output.
        expected_params: Schema mapping parameter name to its type info.

    Returns:
        A new dict with coerced values.
    """
    coerced: Dict[str, Any] = {}

    for key, val in parameters.items():
        if (
            key in expected_params
            and expected_params[key].get("type") == "number"
            and isinstance(val, int)
            and not isinstance(val, bool)
        ):
            coerced[key] = float(val)
        elif isinstance(val, str):
            coerced[key] = val.strip()
        else:
            coerced[key] = val

    return coerced


def find_expected_params(
    raw_functions: List[Dict[str, Any]], fn_name: str
) -> Dict[str, Any]:
    """Look up the parameter schema for a given function name.

    Args:
        raw_functions: List of raw function definitions.
        fn_name: Name of the function to search for.

    Returns:
        The function's ``parameters`` schema, or an empty dict if the
        function is not found.
    """
    for fn in raw_functions:
        if fn.get("name") == fn_name:
            result: Dict[str, Any] = fn.get("parameters", {})
            return result

    return {}


def process_prompt(
    prompt_text: str, cache: MaskCache, raw_functions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Resolve a single prompt into a validated function call.

    Generates constrained JSON for the prompt, coerces parameter
    types, validates the result against ``FunctionCallResult``, and
    prints it. On failure, an error is reported on stderr and the
    program exits.

    Args:
        prompt_text: The natural-language prompt to process.
        cache: Precomputed mask/model data used for generation.
        raw_functions: List of raw function definitions.

    Returns:
        The validated function call result as a dict.
    """
    raw_json_string = generate_constrained_json(prompt_text, cache)

    try:
        extracted_dict = json.loads(raw_json_string)

        fn_name = extracted_dict.get("name")
        expected_params = find_expected_params(raw_functions, fn_name)

        if "parameters" in extracted_dict:
            extracted_dict["parameters"] = coerce_parameter_types(
                extracted_dict["parameters"], expected_params
            )

        final_data = {
            "prompt": prompt_text,
            "name": extracted_dict["name"],
            "parameters": extracted_dict["parameters"],
        }

        result = FunctionCallResult(**final_data)

        print()
        print_json(data=final_data)

        return result.model_dump()

    except ValidationError as e:
        print(
            "Validation failed: output data is invalid or incomplete.",
            file=sys.stderr,
        )
        print(e.errors()[0])
        sys.exit(0)

    except Exception as e:
        print(f"An unexpected error occured.\nDetails: {e}", file=sys.stderr)
        sys.exit(0)


def run_all_prompts(
    raw_prompts: List[Dict[str, Any]],
    cache: MaskCache,
    raw_functions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Process every prompt and collect the resulting function calls.

    Args:
        raw_prompts: List of dicts each containing a "prompt" key.
        cache: Precomputed mask/model data used for generation.
        raw_functions: List of raw function definitions.

    Returns:
        List of validated function call results, one per prompt.
    """
    final_results_list: List[Dict[str, Any]] = []

    for prompt in raw_prompts:
        prompt_text: str = prompt["prompt"]
        print(f"\nPrompt: {prompt_text}")

        result = process_prompt(prompt_text, cache, raw_functions)
        final_results_list.append(result)

    return final_results_list


def print_elapsed_time(start_time: datetime) -> None:
    """Print the time elapsed since ``start_time`` as minutes and seconds.

    Args:
        start_time: The reference start time.
    """
    elapsed_time = datetime.now() - start_time
    total_seconds = elapsed_time.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    print(f"\nAll done in {minutes}m {seconds}s!")
