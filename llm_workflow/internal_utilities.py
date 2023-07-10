
"""Helper functions and classes."""
import inspect
import datetime
import hashlib
from collections.abc import Callable
import tenacity



class Timer:
    """Provides way to time the duration of code within the context manager."""

    def __enter__(self):
        self._start = datetime.datetime.now()
        return self

    def __exit__(self, *args):  # noqa
        self._end = datetime.datetime.now()
        self.interval = self._end - self._start

    def __str__(self):
        return self.formatted(units='seconds', decimal_places=2)

    def formatted(self, units: str = 'seconds', decimal_places: int = 2) -> str:
        """
        Returns a string with the number of seconds that elapsed on the timer. Displays out to
        `decimal_places`.

        Args:
            units:
                format the elapsed time in terms of seconds, minutes, hours
                (currently only supports seconds)
            decimal_places:
                the number of decimal places to display
        """
        if units == 'seconds':
            return f"{self.interval.total_seconds():.{decimal_places}f} seconds"

        raise ValueError("Only suppports seconds.")


def create_hash(value: str) -> str:
    """Based on `value`, returns a hash."""
    # Create a new SHA-256 hash object
    hash_object = hashlib.sha256()
    # Convert the string value to bytes and update the hash object
    hash_object.update(value.encode('utf-8'))
    # Get the hexadecimal representation of the hash
    return hash_object.hexdigest()


def retry_handler(num_retries: int = 3, wait_fixed: int = 1) -> Callable:
    """
    Returns a tenacity callable object that can be used for retrying a function call.

    ```
    r = retry_handler()
    r(
        openai.Completion.create,
        model="text-davinci-003",
        prompt="Once upon a time,"
    )
    ```
    """
    return tenacity.Retrying(
        stop=tenacity.stop_after_attempt(num_retries),
        wait=tenacity.wait_fixed(wait_fixed),
        reraise=True,
    )


def has_property(obj: object, property_name: str) -> bool:
    """
    Returns True if the object has a property (or instance variable) with the name
    `property_name`.
    """
    # if `obj` is itself a function, it will not have any properties
    if inspect.isfunction(obj):
        return False

    return hasattr(obj, property_name) and \
        not callable(getattr(obj.__class__, property_name, None))


def has_method(obj: object, method_name: str) -> bool:
    """Returns True if the object has a method with the name `property_name`."""
    # if `obj` is itself a function, it will not have any properties
    if inspect.isfunction(obj):
        return False
    return hasattr(obj, method_name) and callable(getattr(obj.__class__, method_name, None))
