from app.core.exceptions import EmptyQuery, QueryTooLong

MAX_QUESTION_LENGTH = 2000


def validate_question(question: str) -> str:
    stripped = question.strip() if question else ""
    if not stripped:
        raise EmptyQuery("Question cannot be empty.")
    if len(stripped) > MAX_QUESTION_LENGTH:
        raise QueryTooLong(
            f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters.",
            details={"length": len(stripped), "max": MAX_QUESTION_LENGTH},
        )
    return stripped
