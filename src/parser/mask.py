from typing import Any
import numpy as np
import re

from src.parser.name_resolution import get_allowed_chars


def compute_generation_mask(
        current_str: str, cache: Any,
        vocab_size: int) -> tuple[np.ndarray, str | None]:
    rules = get_allowed_chars(current_str, cache.allowed_fn)
    mask = np.zeros(vocab_size, dtype=bool)

    if len(rules) > 10:
        match = re.search(r'"name"\s*:\s*"([^"]+)', current_str)
        func_name = match.group(1) if match else ""

        params_str = current_str.split('"parameters"')[
            1] if '"parameters"' in current_str else ""

        if params_str:
            in_string = False
            last_structural_colon = -1
            last_structural_comma = -1
            last_structural_brace = -1

            for i, char in enumerate(params_str):
                if char == '"':
                    if i == 0 or params_str[i-1] != '\\':
                        in_string = not in_string
                elif not in_string:
                    if char == ':':
                        last_structural_colon = i
                    elif char == ',':
                        last_structural_comma = i
                    elif char == '}':
                        last_structural_brace = i

            is_inside_value = (
                last_structural_colon > last_structural_comma
                and last_structural_colon > last_structural_brace
            )

            active_key = ""
            if is_inside_value:
                keys_found = re.findall(r'"([^"]+)"\s*:', params_str)
                if keys_found:
                    active_key = keys_found[-1]

            expected_type = cache.param_types.get(
                func_name, {}).get(active_key, "Any")

            param_count = len(re.findall(r'"([^"]+)"\s*:', params_str))
            target_count = cache.func_params.get(func_name, 99)

            if is_inside_value and expected_type == "number":
                mask = cache.p4_numbers_only.copy()

                if param_count == target_count:
                    for i, s in cache.clean_dict_items:
                        if ',' in s:
                            mask[i] = False

            elif is_inside_value and in_string:
                mask = cache.p4_mask.copy()

                if param_count == target_count:
                    for i, s in cache.clean_dict_items:
                        if '",' in s.replace(" ", ""):
                            mask[i] = False

                if active_key == "regex":
                    for i, s in cache.clean_dict_items:
                        if ' ' in s:
                            mask[i] = False

                    if re.search(r'"regex"\s*:\s*"$', params_str):
                        for i, s in cache.clean_dict_items:
                            if not any(s.startswith(c)
                                       for c in ['[', '\\']):
                                mask[i] = False

            elif param_count == target_count:
                clean_str = current_str.strip()
                if clean_str.endswith('"'):
                    return mask, clean_str + "}}"
                elif clean_str.endswith('}'):
                    return mask, clean_str + "}"
                elif clean_str.endswith(','):
                    return mask, clean_str[:-1] + "}}"
                else:
                    mask = cache.p4_no_comma.copy()

            else:
                mask = cache.p4_mask.copy()
                is_expecting_key = (
                    params_str.strip().endswith('{')
                    or params_str.strip().endswith(',')
                )
                if is_expecting_key:
                    for i, s in cache.clean_dict_items:
                        cleaned = s.strip()
                        if not (cleaned.startswith('"') or not cleaned):
                            mask[i] = False

    else:
        for i, s in cache.mini_dict:
            if any(rule.startswith(s) for rule in rules):
                mask[i] = True

    return mask, None
