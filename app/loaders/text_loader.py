from app.core.exceptions import AppError


class TextDecodeError(AppError):
    pass


def load_text(content: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise TextDecodeError("Could not decode text file with utf-8, utf-16, or latin-1.")
