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
    for fn in raw_functions:
        if fn.get("name") == fn_name:
            result: Dict[str, Any] = fn.get("parameters", {})
            return result

    return {}


def process_prompt(
    prompt_text: str, cache: MaskCache, raw_functions: List[Dict[str, Any]]
) -> Dict[str, Any]:
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
        sys.exit(1)

    except Exception as e:
        print(f"An unexpected error occured.\nDetails: {e}", file=sys.stderr)
        sys.exit(1)


def run_all_prompts(
    raw_prompts: List[Dict[str, Any]],
    cache: MaskCache,
    raw_functions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    final_results_list: List[Dict[str, Any]] = []

    for prompt in raw_prompts:
        prompt_text: str = prompt["prompt"]
        print(f"\nPrompt: {prompt_text}")

        result = process_prompt(prompt_text, cache, raw_functions)
        final_results_list.append(result)

    return final_results_list


def print_elapsed_time(start_time: datetime) -> None:
    elapsed_time = datetime.now() - start_time
    total_seconds = elapsed_time.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    print(f"\nAll done in {minutes}m {seconds}s!")
