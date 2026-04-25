from ftfy import fix_text

_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")


def load_text(content: bytes) -> str:
    try:
        return fix_text(content.decode("utf-8"))
    except UnicodeDecodeError:
        pass
    if content.startswith(_UTF16_BOMS):
        try:
            return fix_text(content.decode("utf-16"))
        except UnicodeDecodeError:
            pass
    return fix_text(content.decode("latin-1"))
