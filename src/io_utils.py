import json
import pathlib
import sys
from typing import Any, Dict, List


def load_json_file(filename: str) -> Any:
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
