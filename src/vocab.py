"""Vocabulary loading and filtering utilities for the LLM tokenizer."""

from typing import Any, Dict, List, Tuple
import string

from src.io_utils import load_json_file


def build_vocab_dict(model: Any) -> Dict[int, str]:
    """Build a mapping from token id to token string for the model.

    Args:
        model: LLM wrapper exposing ``get_path_to_vocab_file``.

    Returns:
        Dict mapping each token id to its decoded string, with the
        tokenizer's leading-space marker ("Ġ") replaced by a space.
    """
    vocab_file = model.get_path_to_vocab_file()
    raw_vocab: Dict[str, int] = load_json_file(vocab_file)

    return {v: k.replace("Ġ", " ") for k, v in raw_vocab.items()}


def filter_printable_tokens(
    vocab_dict: Dict[int, str],
) -> Tuple[List[int], List[Tuple[int, str]]]:
    """Keep only tokens made entirely of printable characters.

    Args:
        vocab_dict: Mapping from token id to token string.

    Returns:
        A tuple of (list of valid token ids, list of (id, token) pairs)
        for tokens that are non-empty and fully printable.
    """
    printable_set = set(string.printable)

    valid_ids: List[int] = [
        token_id
        for token_id, token_str in vocab_dict.items()
        if token_str and all(c in printable_set for c in token_str)
    ]

    clean_dict_items: List[Tuple[int, str]] = [
        (k, v)
        for k, v in vocab_dict.items()
        if v and all(c in printable_set for c in v)
    ]

    return valid_ids, clean_dict_items


def compute_vocab_size(model: Any) -> int:
    """Determine the model's vocabulary size from a dummy forward pass.

    Args:
        model: LLM wrapper exposing ``encode`` and
            ``get_logits_from_input_ids``.

    Returns:
        The number of logits returned for a single dummy input, i.e.
        the vocabulary size.
    """
    dummy_input_ids = model.encode("dummy").tolist()[0]
    return len(model.get_logits_from_input_ids(dummy_input_ids))
