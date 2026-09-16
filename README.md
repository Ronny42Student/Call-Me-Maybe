*This project has been created as part of the 42 curriculum by nrajaoar.*

## Description

**call me maybe** turns a natural-language question (e.g. "What is the sum of 2 and 3?") into a structured, valid function call:

```
Input:  "What is the sum of 2 and 3?"
Output: {"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

The model (`Qwen/Qwen3-0.6B`) is too small to reliably produce valid JSON on its own by prompting alone. This project instead uses **constrained decoding**: at every step, the model's logits are masked so that only tokens compatible with valid JSON and the target function schema can be picked. This guarantees 100% valid, parseable output no matter how small the model is.

## Instructions

```bash
uv sync
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

By default, input files are read from `data/input/` and the result is written to `data/output/function_calling_results.json`.

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

---

## Step-by-step simulation (function by function)

Context used for every step below:

```
prompt_text = "What is the sum of 2 and 3?"

cache.allowed_fn   = ["fn_add_numbers", "fn_greet"]
cache.func_params  = {"fn_add_numbers": 2, "fn_greet": 1}
cache.param_types  = {"fn_add_numbers": {"a": "number", "b": "number"},
                       "fn_greet": {"name": "string"}}
```

### 1) `build_full_prompt` — build the text sent to the model

This function turns the raw prompt and the function definitions into one big text block the model can read, including the rules it must follow (only valid JSON, no literal regex matches, etc.).

```
INPUT  : prompt_text, cache
OUTPUT : one text block containing the rules, the schemas, and the question

prompt = build_full_prompt(prompt_text, cache)

=> prompt =
   "System: output ONLY valid JSON matching these schemas...
    Schemas: [fn_add_numbers, fn_greet]
    Rule: for the regex field, never output a literal match
    User: What is the sum of 2 and 3?
    Tool Call:"
```

### 2) Initialization — before the loop, in `generate_constrained_json`

Before the loop starts, the prompt is tokenized once, and the JSON is seeded with a fixed opening prefix that every valid function call must start with.

```
input_ids  = cache.model.encode(prompt).tolist()[0]   # about a hundred tokens
vocab_size = len(cache.model.get_logits_from_input_ids(input_ids))  # 150000

prefix      = '{"name":"'
current_str = '{"name":"'
input_ids.extend(cache.model.encode(prefix).tolist()[0])

bridge_injected = False
max_tokens       = 150
```

### 3) `resolve_function_name` — round 1: still 2 possible functions

This helper checks whether the partial function name written so far is unique enough to be completed directly. Here, both function names still match, so nothing can be decided yet.

```
INPUT  : current_str = '{"name":"', cache.allowed_fn, prefix
OUTPUT : None (no decision possible, still several candidates)

after_prefix              = current_str.split(prefix)[1] = ""
functions starting with
""                        = ["fn_add_numbers", "fn_greet"]   (both)

=> 2 candidates -> return None
```

The bridge-injection check (in `generate_constrained_json`) also fails, so generation falls through to masked sampling:
```
current_str.endswith('"')       -> True
not bridge_injected             -> True
prefix in current_str           -> True
len(current_str) > len(prefix)  -> False
=> overall condition FALSE: no injection, continue to masking
```

### 4) `get_allowed_chars` — round 1: which characters are still valid?

This function lists the literal continuations that are still consistent with the allowed function names, given what has been written so far.

```
INPUT  : current_str = '{"name":"', allowed_names
OUTPUT : list of string continuations still possible

prefix = '{"name":"'
len(current_str) < len(prefix) ?  -> False (current_str equals prefix exactly)

after_prefix = current_str[len(prefix):] = ""
'"' in after_prefix ?              -> False

=> return [name[0:] + '"' for name in allowed_names if name.startswith("")]
   = ["fn_add_numbers", "fn_greet"]   (both names are still open)
```

### 5) `compute_generation_mask` — round 1: picking one character

This function turns the allowed continuations into an actual boolean mask over the vocabulary, so invalid tokens can be excluded before the model picks the next one.

```
INPUT  : current_str = '{"name":"', cache, vocab_size
OUTPUT : mask (True = allowed), forced_close = None

rules = get_allowed_chars(...) = ["fn_add_numbers", "fn_greet"]
len(rules) > 10 ?  -> False (only 2 rules)

=> "few rules" branch: use cache.mini_dict
   for each (i, s) in mini_dict:
       if any rule starts with s -> mask[i] = True

   e.g. mask["f"] = True, mask["fn"] = True, mask["fn_"] = True,
        mask["fn_a"] = True, mask["fn_g"] = True, everything else = False
```

Back in `generate_constrained_json`:
```
logits = np.array(cache.model.get_logits_from_input_ids(input_ids))
logits[~mask] = -np.inf              # forbidden tokens set to -infinity

best_id = int(np.argmax(logits))     # say best_id = 501
current_str += cache.vocab_dict.get(501, "")
# current_str = '{"name":"fn_a'

input_ids.append(501)
current_str.endswith('"') -> False   # ends with 'a', loop continues
```

### 6) `resolve_function_name` — round 2: only 1 function left

Now that only one function name is still compatible with what has been generated, the rest of the name is written in one go instead of being sampled token by token.

```
INPUT  : current_str = '{"name":"fn_a'
OUTPUT : the rest of the name, appended directly (no more guessing needed)

after_prefix   = "fn_a"
possible_names = [n for n in allowed_fn if n.startswith("fn_a")]
               = ["fn_add_numbers"]        (only one!)

len(possible_names) == 1 and possible_names[0] != after_prefix  -> True

=> return "fn_add_numbers"[len("fn_a"):] + '"'
        = "fn_add_numbers"[4:] + '"'
        = 'dd_numbers"'

remainder = 'dd_numbers"'
current_str += remainder
# current_str = '{"name":"fn_add_numbers"'
```

### 7) `inject_parameters_bridge` — moving into the parameters object

Once the function name is resolved and closed, this function appends the fixed JSON bridge into the parameters object, and either closes the call immediately (zero-parameter functions) or prepares a smaller, function-scoped prompt.

```
INPUT  : current_str = '{"name":"fn_add_numbers"', input_ids, cache, prompt_text
OUTPUT : current_str with "parameters" appended, updated input_ids, is_finished

Check before the call (in generate_constrained_json):
current_str.endswith('"')       -> True
not bridge_injected             -> True
prefix in current_str           -> True
len(current_str) > len(prefix)  -> True
=> all true: call inject_parameters_bridge

bridge = ',"parameters":{'
current_str += bridge
# current_str = '{"name":"fn_add_numbers","parameters":{'

func_name = current_str.split('"name":"')[1].split('"')[0] = "fn_add_numbers"
cache.func_params.get("fn_add_numbers", 99) = 2   # not 0 -> no immediate close

active_schema = the fn_add_numbers definition found in cache.raw_functions

=> a tiny_prompt is rebuilt (build_function_scoped_prompt), focused only on
   fn_add_numbers, and input_ids is re-encoded from it

is_finished = False
bridge_injected = True
```

### 8) `build_function_scoped_prompt` — the refocused prompt

Once the function is known, a smaller prompt is built with only that function's schema, so the model no longer has to consider the other functions while generating the parameters.

```
INPUT  : prompt_text, current_str, active_schema (fn_add_numbers)
OUTPUT : a smaller prompt, scoped to a single function

tiny_schema = {"name": "fn_add_numbers", "description": "...",
               "parameters": {"a": {"type": "number"}, "b": {"type": "number"}}}

tiny_prompt =
   "System: output JSON matching THIS schema: {tiny_schema}
    User: What is the sum of 2 and 3?
    Tool Call: {"name":"fn_add_numbers","parameters":{"
```

### 9) `compute_generation_mask` — entering `"parameters":{`

Here the mask function has to figure out, purely from the partial JSON text, whether the model is about to write a key, a string value, or a numeric value.

```
INPUT  : current_str now contains '...,"parameters":{'
OUTPUT : we now expect a KEY (e.g. "a")

rules = get_allowed_chars(...) -> many possibilities (more than 10)
=> falls into the detailed branch (structural analysis of the JSON so far)

func_name  = "fn_add_numbers"   (found with a regex on '"name":"...')
params_str = ':{'               (everything after '"parameters"')

Scanning params_str character by character, outside of any string:
   i=0 (':') -> last_structural_colon = 0
   i=1 ('{') -> last_structural_brace = 1
                last_structural_comma stays -1

is_inside_value = (last_structural_colon > last_structural_comma
                    and last_structural_colon > last_structural_brace)
                = (0 > -1 and 0 > 1)
                = (True and False)
                = False

active_key = ""    # no key found yet

param_count  = len(re.findall(r'"([^"]+)"\s*:', params_str)) = 0
target_count = cache.func_params.get("fn_add_numbers", 99)   = 2

param_count (0) != target_count (2) and is_inside_value = False
=> "expecting a key" branch:
   mask = cache.p4_mask.copy()
   params_str.strip() ends with '{' -> is_expecting_key = True
   keep only tokens that start with '"' (opening a new key)

=> the model will now choose to write "a"
```

### 10) Full trace to the end (quick step summary)

```
'{"name":"fn_add_numbers","parameters":{'
  |
  | -> compute_generation_mask: KEY mode -> writes '"a":'
  v
'{"name":"fn_add_numbers","parameters":{"a":'
  |
  | -> is_inside_value = True, expected_type = "number"
  |    -> mask = cache.p4_numbers_only.copy()  (only 0-9 . - , } allowed)
  |    -> writes '2.0'
  v
'{"name":"fn_add_numbers","parameters":{"a":2.0'
  |
  | -> param_count(1) != target_count(2) -> back to "expecting a key" mode
  |    -> writes ',"b":'
  v
'{"name":"fn_add_numbers","parameters":{"a":2.0,"b":'
  |
  | -> NUMBER mode -> writes '3.0'
  v
'{"name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0'
  |
  | -> param_count(2) == target_count(2)
  |    clean_str ends in a digit (not '"', '}' or ',')
  |    -> mask = cache.p4_no_comma.copy()  (only allow closing now)
  |    -> the model writes '}'
  v
'{"name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}'
  |
  | -> param_count(2) == target_count(2), clean_str ends with '}'
  |    -> forced_close = clean_str + "}"   (forced closing, no masking needed)
  v
FINAL RESULT:
'{"name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}}'
```

### 11) `process_prompt` — final validation

The raw generated JSON string is parsed, its parameter types are coerced to match the schema, and the result is validated with Pydantic before being written to the output file. This is the safety net that catches any unexpected shape even though constrained decoding should never produce invalid output.

```
INPUT  : raw_json_string = '{"name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}}'
OUTPUT : a validated dict, ready to write to the output file

extracted_dict = json.loads(raw_json_string)
fn_name        = "fn_add_numbers"
expected_params = {"a": {"type": "number"}, "b": {"type": "number"}}

extracted_dict["parameters"] = coerce_parameter_types(
    {"a": 2.0, "b": 3.0}, expected_params)
# already floats -> unchanged

final_data = {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
}

result = FunctionCallResult(**final_data)   # Pydantic validation passes
=> written to data/output/function_calling_results.json
```

### 12) Edge case: a function with no parameters (`inject_parameters_bridge`)

Some functions take zero parameters. In that case the bridge function closes the call immediately, without ever building a scoped prompt.

```
INPUT  : current_str = '{"name":"fn_ping"', func_params["fn_ping"] = 0
OUTPUT : immediate closing, no tiny_prompt built

bridge = ',"parameters":{'
current_str += bridge
# current_str = '{"name":"fn_ping","parameters":{'

func_name = "fn_ping"
cache.func_params.get("fn_ping", 99) == 0   -> True

current_str += "}}"
# current_str = '{"name":"fn_ping","parameters":{}}'

return current_str, input_ids, True   # is_finished = True -> the loop stops
```

---

## Design decisions

- As soon as a single function name remains possible, `resolve_function_name` writes it in one go instead of sampling it token by token — this saves generation steps and removes any risk of the model deviating.
- A scoped prompt (`build_function_scoped_prompt`) is rebuilt once the function is known, so the model only ever sees the schema that is actually relevant.
- Once `param_count == target_count`, generation forces the JSON closed (`}}`) instead of letting the model keep guessing — this guarantees termination and validity.
- The result is always re-validated with Pydantic (`FunctionCallResult`) before being saved, and parameter types are coerced when needed (`coerce_parameter_types`), e.g. turning an `int` into a `float` for a `"number"` field.

## Testing strategy

Every function (`build_full_prompt`, `resolve_function_name`, `get_allowed_chars`, `inject_parameters_bridge`, `build_function_scoped_prompt`, `compute_generation_mask`, `process_prompt`) was traced by hand as shown in the simulation above, covering edge cases: an empty string, a single remaining candidate, a zero-parameter function, the last expected parameter being reached, and parameter types needing coercion.

## Resources

- [JSON specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259)
- [Pydantic documentation](https://docs.pydantic.dev/)

**AI usage:** AI was used to help write the code's docstrings and to lay out the step-by-step simulations in this README. All code was read and understood before being used.