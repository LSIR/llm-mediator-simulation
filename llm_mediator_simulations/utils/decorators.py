"""Utility decorators."""


def retry(attempts=5, verbose=False):
    """Decorator to retry a function a given number of times before failing.

    Args:
        attempts (int): The number of times to attempt calling the function before failing.
        verbose (bool): Whether to print the reason for each failure.
    """

    def decorator(fn):
        def wrapper(*args, **kwargs):
            for _ in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if verbose:
                        print(f"Failed: {e}")
            raise RuntimeError(f"Function {fn.__name__} failed {attempts} times.")

        return wrapper

    return decorator
