import time
from functools import wraps

from TextTranslation.src.common.logger import get_logger

logger = get_logger()


def timeit(func):
    @wraps(func)
    def _timeit(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.debug(f'function {func.__name__} took {total_time:.3f} seconds')
        return result

    return _timeit
