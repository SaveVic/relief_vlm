"""
relief.models
=============
Replicate endpoint registry and input-name mapping.

The spec table is the single source of input names, and `build_model_input()`
reads from it. A wrong key fails silently: the model just receives a request
without images or without a system prompt and still answers.

Each Replicate endpoint has its own input schema; check the "API" tab on the
model page before adding an entry. Things to verify: the image input name, the
system prompt name, the output length limit name, and that limit's default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    """Input names for one Replicate endpoint."""

    slug: str
    image_key: str            # input name for the image list
    system_key: str           # input name for the system prompt
    max_tokens_key: str       # input name for the output length limit
    max_tokens: int           # value used by the experiment, not the endpoint default
    supports_temperature: bool = True
    extra_input: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


# OpenAI reasoning models do not accept `temperature`; the closest substitute
# is `reasoning_effort`, set via `extra_input`.
REASONING_PREFIXES = ("openai/gpt-5", "openai/o1", "openai/o3", "openai/o4")


MODELS: Dict[str, ModelSpec] = {
    "openai/gpt-4.1": ModelSpec(
        "openai/gpt-4.1", "image_input", "system_prompt",
        "max_completion_tokens", 12000,
        note="default max_completion_tokens 4096 truncates the four-layer record",
    ),
    "openai/gpt-4.1-mini": ModelSpec(
        "openai/gpt-4.1-mini", "image_input", "system_prompt",
        "max_completion_tokens", 12000,
        note="cheap counterpart of gpt-4.1; tests whether enhancement helps "
             "weaker models more",
    ),
    "openai/gpt-4o": ModelSpec(
        "openai/gpt-4o", "image_input", "system_prompt",
        "max_completion_tokens", 12000,
    ),
    "openai/gpt-4o-mini": ModelSpec(
        "openai/gpt-4o-mini", "image_input", "system_prompt",
        "max_completion_tokens", 12000,
    ),
    "google/gemini-3-flash": ModelSpec(
        "google/gemini-3-flash", "images", "system_instruction",
        "max_output_tokens", 16000,
        note="different family; also has thinking_level. Note the unusual "
             "system prompt and length limit input names",
    ),
    "qwen/qwen3-7-plus": ModelSpec(
        "qwen/qwen3-7-plus", "image", "system_prompt",
        "max_tokens", 16000,
        note="input is named 'image' (singular) but accepts a list of images; "
             "default max_tokens 2048 is too short",
    ),
    "ibm-granite/granite-vision-3.3-2b": ModelSpec(
        "ibm-granite/granite-vision-3.3-2b", "images", "system_prompt",
        "max_tokens", 8000,
        note="2B model, default max_tokens 512. A high parse failure rate "
             "is an experimental finding, not a bug",
    ),
}


def base_name(model: str) -> str:
    """'owner/name:version' -> 'owner/name'."""
    return model.split(":", 1)[0]


def get_spec(model: str) -> ModelSpec:
    name = base_name(model)
    if name not in MODELS:
        raise KeyError(
            f"model {name!r} is not registered. Add a ModelSpec in "
            f"relief/models.py following the API tab on replicate.com. "
            f"Registered: {sorted(MODELS)}"
        )
    return MODELS[name]


def build_model_input(
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: List[str],
    temperature: float,
    max_tokens: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the input dict using the input names of the given endpoint."""
    spec = get_spec(model)
    payload: Dict[str, Any] = {
        "prompt": user_prompt,
        spec.system_key: system_prompt,
        spec.image_key: images,
        spec.max_tokens_key: max_tokens if max_tokens is not None else spec.max_tokens,
    }
    if spec.supports_temperature and not base_name(model).startswith(REASONING_PREFIXES):
        payload["temperature"] = temperature
    payload.update(spec.extra_input)
    if extra:
        payload.update(extra)
    return payload


def spec_table() -> str:
    """Table of input names, to check before running a new model."""
    head = f"{'model':<36}{'images':<14}{'system prompt':<20}{'output limit':<24}{'value'}"
    lines = [head, "-" * len(head)]
    for spec in MODELS.values():
        lines.append(
            f"{spec.slug:<36}{spec.image_key:<14}{spec.system_key:<20}"
            f"{spec.max_tokens_key:<24}{spec.max_tokens}"
        )
    lines += [
        "",
        "A complete four-layer record takes about 4-5 thousand output tokens.",
        "Endpoints left at their default (4096 for gpt-4.1, 2048 for qwen,",
        "512 for granite) get cut off mid-JSON and are recorded as parse",
        "failures for the wrong reason.",
    ]
    return "\n".join(lines)
