from typing import Any
import json


def build_full_prompt(prompt_text: str, cache: Any) -> str:
    optimized_schemas = []
    for f in cache.raw_functions:
        optimized_schemas.append({
            "name": f["name"],
            "description": f.get("description", ""),
            "parameters": f.get("parameters", {})
        })

    schema_hints = json.dumps(optimized_schemas, separators=(',', ':'))

    prompt = (
        f"System: You are a strict API. Output ONLY valid JSON matching "
        f"these schemas: {schema_hints}\n"
        r"Rule: For the regex field, NEVER output literal matches. "
        r"Always use proper regex sets "
        r"(e.g. '[aeiouAEIOU]', '[0-9]+', '\\bword\\b'). "
        r"For replacement, if asked for a character (e.g. asterisks), "
        r"output EXACTLY ONE character (e.g. '*')."
        "\n"
        f"User: {prompt_text}\n"
        "Tool Call: "
    )

    return prompt


def build_function_scoped_prompt(
        prompt_text: str, current_str: str,
        active_schema: dict[str, Any]) -> str:
    tiny_schema = json.dumps(
        [{
            "name": active_schema["name"],
            "description": active_schema.get("description", ""),
            "parameters": active_schema.get("parameters", {})
        }],
        separators=(',', ':')
    )

    tiny_prompt = (
        f"System: Output valid JSON matching this schema: "
        f"{tiny_schema}\n"
        r"Rule: For the regex field, NEVER output "
        r"literal matches. "
        r'Always use proper regex sets '
        r'(e.g. "[aeiouAEIOU]", "[0-9]+", "\\bword\\b"). '
        r"For replacement, if asked for a character "
        r"(e.g. asterisks), output EXACTLY ONE character "
        r"(e.g. '*')."
        "\n"
        f"User: {prompt_text}\n"
        f"Tool Call: {current_str}"
    )

    return tiny_prompt
