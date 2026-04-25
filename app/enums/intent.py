from enum import StrEnum


class Intent(StrEnum):
    GREETING = "GREETING"
    FAREWELL = "FAREWELL"
    RAG_QUERY = "RAG_QUERY"
