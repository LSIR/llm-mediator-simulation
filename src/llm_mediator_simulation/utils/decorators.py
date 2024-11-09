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
                        print(f"Failed: {type(e).__name__}: {e}")
                        print (fn.__name__)
            raise RuntimeError(f"Function {fn.__name__} failed {attempts} times.")

        return wrapper

    return decorator


BENCHMARKS: dict[str, list[float]] = {}


def benchmark(name: str, verbose=True):
    """Decorator to benchmark a function call.

    Args:
        name (str): The name of the function to benchmark.
        verbose (bool): Whether to print the benchmark call times live.
    """
    from time import time

    def decorator(fn):
        def wrapper(*args, **kwargs):
            start = time()
            result = fn(*args, **kwargs)
            end = time()
            if verbose:
                print(f"{name} took {end - start} seconds.")
            BENCHMARKS.setdefault(name, []).append(end - start)
            return result

        return wrapper

    return decorator


def print_benchmarks(benchmarks: dict[str, list[float]] | None = None) -> None:
    """Pretty print benchmark results.

    Args:
        benchmarks (dict[str, list[float]] | None): The benchmarks to print. If None, print the global benchmarks.
    """

    from rich.console import Console
    from rich.table import Table

    table = Table()

    table.add_column("Function")
    table.add_column("Total time (s)")
    table.add_column("Average time (s)")
    table.add_column("Standard deviation (s)")

    benchmarks = benchmarks or BENCHMARKS

    for name, times in benchmarks.items():
        table.add_row(
            name,
            str(sum(times)),
            str(sum(times) / len(times)),
            str(
                (
                    sum((time - sum(times) / len(times)) ** 2 for time in times)
                    / len(times)
                )
                ** 0.5
            ),
        )

    console = Console()
    console.print(table)
