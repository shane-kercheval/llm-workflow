"""Contains definitions of exceptions/errors."""

class RequestError(Exception):
    """Class that wraps an error when using requests.get()."""

    def __init__(self, status_code: int, reason: str) -> None:
        self.status_code = status_code
        self.reason = reason
