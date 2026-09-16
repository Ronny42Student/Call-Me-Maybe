"""File I/O helpers for reading and writing JSON data."""

import json
import pathlib
import sys
from typing import Any, Dict, List


def load_json_file(filename: str) -> Any:
    """Load and parse a JSON file, exiting on any error.

    Args:
        filename: Path to the JSON file to load.

    Returns:
        The parsed JSON content.

    Raises:
        SystemExit: If the file does not exist, is not readable, or
            contains invalid JSON.
    """
    if not pathlib.Path(filename).is_file():
        raise SystemExit(
            f"This file {filename} is not found, or it is a directory."
        )

    try:
        with open(filename, "r") as file:
            json_data = json.load(file)

    except json.JSONDecodeError:
        print(
            "The JSON file is invalid or corrupted. "
            "A required key or value is missing.",
            file=sys.stderr,
        )
        sys.exit(1)

    except PermissionError:
        print(f"Permission denied in this file {filename}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"An error occured.\nDetails: {e}", file=sys.stderr)
        sys.exit(1)

    return json_data


def write_output_file(
    output_path: str, results: List[Dict[str, Any]]
) -> None:
    """Write results to a JSON file, creating parent directories if needed.

    Args:
        output_path: Destination path for the JSON output file.
        results: List of result dicts to serialize.

    Raises:
        SystemExit: If the path is invalid or the file cannot be written.
    """
    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.touch(exist_ok=True)

    if not output_file.is_file():
        print(
            f"This file '{output_path}' is not found, or it is a directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(output_path, "w") as file:
            json.dump(results, file, indent=4)

    except PermissionError:
        print(f"Permission denied in this file {output_path}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"An error occured.\nDetails: {e}", file=sys.stderr)
        sys.exit(1)
