"""Entry point for the Call Me Maybe function calling tool."""

import argparse
import sys
from datetime import datetime
from typing import Any, Dict, List

from llm_sdk import Small_LLM_Model

from src.io_utils import load_json_file, write_output_file
from src.masks import build_mask_cache
from src.pipeline import print_elapsed_time, run_all_prompts


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the function calling tool.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="CallMeMaybe",
        description="Call Me Maybe: LLM Function Calling Tool",
        usage="uv run python -m src [--functions_definition "
        "<function_definition_file>] [--input <input_file>] "
        "[--output <output_file>]",
    )

    parser.add_argument(
        "--functions_definition",
        metavar="",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to JSON file containing the functions definitions.",
    )

    parser.add_argument(
        "--input",
        metavar="",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the file containing the prompts.",
    )

    parser.add_argument(
        "--output",
        metavar="",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to the JSON output file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace Model ID",
    )

    return parser.parse_args()


def main() -> None:
    """Run the full function calling pipeline end to end.

    Loads the function definitions and prompts, builds the constrained
    decoding cache, resolves every prompt to a function call, and
    writes the results to the output file.
    """
    args = parse_arguments()

    raw_functions: List[Dict[str, Any]] = load_json_file(
        args.functions_definition
    )
    raw_prompts: List[Dict[str, Any]] = load_json_file(args.input)

    model = Small_LLM_Model(model_name=args.model)

    start_time = datetime.now()

    cache = build_mask_cache(model, raw_functions)

    final_results_list = run_all_prompts(raw_prompts, cache, raw_functions)

    write_output_file(args.output, final_results_list)

    print_elapsed_time(start_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
