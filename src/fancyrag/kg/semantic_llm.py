"""Structured-output LLM adapter for semantic extraction."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    from neo4j_graphrag.exceptions import LLMGenerationError
    from neo4j_graphrag.llm.base import LLMInterface
    from neo4j_graphrag.llm.types import LLMResponse
except ModuleNotFoundError:  # pragma: no cover - minimal environments only
    class LLMGenerationError(RuntimeError):
        """Fallback LLM error used when neo4j_graphrag is unavailable."""


    class LLMInterface:  # type: ignore[no-redef]
        """Minimal LLM interface placeholder for structured semantic output."""

        def __init__(
            self,
            *_args,
            model_name: str | None = None,
            model_params: Mapping[str, Any] | None = None,
            **_kwargs,
        ) -> None:
            self.model_name = model_name
            self.model_params = dict(model_params or {})


    @dataclass
    class LLMResponse:  # type: ignore[no-redef]
        """Fallback response container when neo4j_graphrag is unavailable."""

        content: str | None = None

from cli.openai_client import OpenAIClientError, SharedOpenAIClient


_FENCE_PATTERN = re.compile(r"^```[ \t]*([^\n`]*)\s*(.*?)\s*```$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_PATTERN.match(stripped)
    if not match:
        return stripped
    return match.group(2).strip()


def _extract_content(raw_response: Any) -> str:
    payload = raw_response
    if hasattr(raw_response, "model_dump"):
        payload = raw_response.model_dump()
    elif hasattr(raw_response, "to_dict"):
        payload = raw_response.to_dict()

    if isinstance(payload, Mapping):
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

    if isinstance(payload, Mapping):
        outputs = payload.get("output") or []
    else:
        outputs = getattr(payload, "output", None) or []
    for output in outputs:
        contents = output.get("content") if isinstance(output, Mapping) else getattr(output, "content", None)
        contents = contents or []
        for item in contents:
            item_text = item.get("text") if isinstance(item, Mapping) else getattr(item, "text", None)
            if isinstance(item_text, str) and item_text.strip():
                return item_text
    return ""


def _looks_like_format_error(exc: OpenAIClientError) -> bool:
    message = str(exc).lower()
    return "response_format" in message or "json_schema" in message or "text.format" in message


class StructuredSemanticLLM(LLMInterface):
    """LLM adapter that enforces structured output for semantic extraction."""

    def __init__(self, client: SharedOpenAIClient, settings: Any, *, schema: dict[str, Any]) -> None:
        super().__init__(
            model_name=settings.chat_model,
            model_params={"temperature": getattr(settings, "temperature", 0.0), "max_tokens": 512},
        )
        self._client = client
        self._settings = settings
        self._schema = schema

    def _build_messages(
        self,
        input_text: str,
        message_history: Sequence[Mapping[str, str]] | None,
        system_instruction: str | None,
    ) -> list[Mapping[str, str]]:
        messages: list[Mapping[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        if message_history:
            messages.extend(message_history)
        messages.append({"role": "user", "content": input_text})
        return messages

    def _format_payload(self) -> dict[str, Any] | None:
        response_format = getattr(self._settings, "semantic_response_format", "json_schema")
        if response_format == "off":
            return None
        if response_format == "json_object":
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "name": "neo4j_graph",
            "schema": self._schema,
            "strict": bool(getattr(self._settings, "semantic_schema_strict", True)),
        }

    def invoke(
        self,
        input: str,
        message_history: Sequence[Mapping[str, str]] | None = None,
        system_instruction: str | None = None,
    ) -> LLMResponse:
        messages = self._build_messages(input, message_history, system_instruction)
        text_format = self._format_payload()
        extra_params: dict[str, Any] = {}
        if text_format is not None:
            extra_params["text"] = {"format": text_format}
        try:
            result = self._client.chat_completion(
                messages=messages,
                temperature=self.model_params.get("temperature", 0.0),
                max_tokens=self.model_params.get("max_tokens", 512),
                **extra_params,
            )
        except OpenAIClientError as exc:
            if (
                text_format is not None
                and text_format.get("type") == "json_schema"
                and _looks_like_format_error(exc)
            ):
                fallback = {"format": {"type": "json_object"}}
                result = self._client.chat_completion(
                    messages=messages,
                    temperature=self.model_params.get("temperature", 0.0),
                    max_tokens=self.model_params.get("max_tokens", 512),
                    text=fallback,
                )
            else:
                raise LLMGenerationError(str(exc)) from exc

        content = _strip_code_fence(_extract_content(result.raw_response))
        if not content:
            raise LLMGenerationError("OpenAI returned an empty response")
        return LLMResponse(content=content)

    async def ainvoke(
        self,
        input: str,
        message_history: Sequence[Mapping[str, str]] | None = None,
        system_instruction: str | None = None,
    ) -> LLMResponse:
        return await asyncio.to_thread(self.invoke, input, message_history, system_instruction)


__all__ = ["StructuredSemanticLLM"]
