from dataclasses import dataclass

import tiktoken

from app.core.config import Settings

# Ordered from coarsest to finest — splitter tries each separator in turn, using larger
# units (paragraphs) before falling back to sentences, words, and characters.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(slots=True)
class TextChunk:
    index: int
    text: str
    token_count: int


class RecursiveSplitter:
    def __init__(self, settings: Settings) -> None:
        self._chunk_size = settings.chunk_size_tokens
        self._overlap = settings.chunk_overlap_tokens
        try:
            self._encoding = tiktoken.encoding_for_model(settings.openai_embedding_model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def split(self, text: str) -> list[TextChunk]:
        if not text.strip():
            return []

        segments = self._recursive_split(text, 0)
        merged = self._merge_segments(segments)
        return [
            TextChunk(index=i, text=segment, token_count=self._count_tokens(segment))
            for i, segment in enumerate(merged)
        ]

    def _recursive_split(self, text: str, separator_index: int) -> list[str]:
        if self._count_tokens(text) <= self._chunk_size:
            return [text]

        if separator_index >= len(_SEPARATORS):
            return self._split_by_tokens(text)

        separator = _SEPARATORS[separator_index]
        if separator == "":
            return self._split_by_tokens(text)

        parts = text.split(separator)
        segments: list[str] = []
        for part in parts:
            if not part:
                continue
            piece = part if separator_index == 0 else separator + part if segments else part
            if self._count_tokens(piece) > self._chunk_size:
                segments.extend(self._recursive_split(piece, separator_index + 1))
            else:
                segments.append(piece)
        return segments

    def _split_by_tokens(self, text: str) -> list[str]:
        tokens = self._encoding.encode(text)
        segments: list[str] = []
        stride = max(1, self._chunk_size - self._overlap)
        for start in range(0, len(tokens), stride):
            window = tokens[start : start + self._chunk_size]
            if not window:
                break
            segments.append(self._encoding.decode(window))
        return segments

    def _merge_segments(self, segments: list[str]) -> list[str]:
        merged: list[str] = []
        buffer = ""
        buffer_tokens = 0

        for segment in segments:
            segment_tokens = self._count_tokens(segment)

            if buffer_tokens + segment_tokens <= self._chunk_size:
                buffer = (buffer + "\n" + segment).strip() if buffer else segment
                buffer_tokens = self._count_tokens(buffer)
                continue

            if buffer:
                merged.append(buffer)

            if segment_tokens > self._chunk_size:
                merged.extend(self._split_by_tokens(segment))
                buffer = ""
                buffer_tokens = 0
                continue

            buffer = self._build_overlap_prefix(merged) + segment if merged else segment
            buffer_tokens = self._count_tokens(buffer)

        if buffer:
            merged.append(buffer)

        return merged

    def _build_overlap_prefix(self, merged: list[str]) -> str:
        if not merged or self._overlap <= 0:
            return ""
        last = merged[-1]
        tokens = self._encoding.encode(last)
        if len(tokens) <= self._overlap:
            return last + "\n"
        tail = self._encoding.decode(tokens[-self._overlap :])
        return tail + "\n"

    def _count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))
