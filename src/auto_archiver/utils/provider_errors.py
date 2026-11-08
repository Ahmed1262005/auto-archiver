class ProviderError(RuntimeError):
    """An ad-library API answered with an error envelope or a page that breaks its documented shape.

    `schema=True` means the response itself is malformed (retrying will not help until the parser is fixed).
    Callers with their own retry policy classify on `status` and `code`.
    """

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None, schema: bool = False):
        super().__init__(message)
        self.status, self.code, self.schema = status, code, schema
