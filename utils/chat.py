"""LLM backed assistant for Q&A with graceful fallbacks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from ai_features import answer_question

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class LLMAssistant:
    """Wrapper around OpenAI or local heuristics for answering questions."""

    model: str = "gpt-3.5-turbo"
    history: List[ChatMessage] = field(default_factory=list)

    def _client(self):  # pragma: no cover - network dependency
        if OpenAI is None:
            return None
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return OpenAI(api_key=api_key)

    def ask(self, question: str, context: str) -> str:
        """Return an answer using the LLM when available."""

        client = self._client()
        self.history.append(ChatMessage(role="user", content=question))
        if client is None:
            return answer_question(question, context)
        try:  # pragma: no cover
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは小規模事業者の経営参謀です。提供された売上分析を踏まえて助言してください。"},
                    {"role": "user", "content": f"分析サマリー:\n{context}"},
                    {"role": "user", "content": question},
                ],
            )
            content = response.choices[0].message.content
            self.history.append(ChatMessage(role="assistant", content=content))
            return content
        except Exception:
            return answer_question(question, context)
