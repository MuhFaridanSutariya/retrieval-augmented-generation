from enum import StrEnum


class FileType(StrEnum):
    PDF = "pdf"
    TEXT = "txt"
    MARKDOWN = "md"

    @classmethod
    def from_extension(cls, extension: str) -> "FileType":
        normalized = extension.lower().lstrip(".")
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unsupported file extension: {extension}")
