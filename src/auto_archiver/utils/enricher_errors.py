class EnricherInputError(ValueError):
    """The media itself can't be analysed (corrupt, no stream, too short ...). Retrying won't help."""

    def __init__(self, message: str, *, reason: str):
        super().__init__(message)
        self.reason = reason


class EnricherServiceError(RuntimeError):
    """The tool or service failed (timeout, 5xx, OOM, crash). Retrying may help."""

    def __init__(self, message: str, *, reason: str):
        super().__init__(message)
        self.reason = reason
