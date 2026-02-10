import functools
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, ParamSpec

import xarray

CACHE_PATH = Path("/data/xarray_cache")

P = ParamSpec("P")


def xarray_cache(func: Callable[P, xarray.Dataset]) -> Callable[P, xarray.Dataset]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> xarray.Dataset:
        # Build module path: e.g. bessie.data._predispatch -> bessie/data/_predispatch
        module_path = Path(func.__module__.replace(".", "/"))

        # Build arguments string from the function signature
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arg_string = "".join(f"{k}={v}" for k, v in bound.arguments.items())

        cache_file = CACHE_PATH / module_path / func.__name__ / f"{arg_string}.netcdf"

        if not cache_file.exists():
            logging.info(f"File not found, creating: {cache_file}")

            result = func(*args, **kwargs)

            cache_file.parent.mkdir(parents=True, exist_ok=True)
            result.to_netcdf(cache_file)
            logging.info(f"Created new file: {cache_file}")

        logging.info(f"Loading cached file: {cache_file}")
        return xarray.open_dataset(cache_file, chunks="auto")

    return wrapper
