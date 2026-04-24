from enum import StrEnum


class DocumentStatus(StrEnum):
    UPLOADED = "UPLOADED"
    INGESTING = "INGESTING"
    READY = "READY"
    FAILED = "FAILED"
