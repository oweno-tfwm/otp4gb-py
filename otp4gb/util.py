import datetime
import multiprocessing
import time
from typing import Callable, Iterator, TypeVar


T = TypeVar("T")
A = TypeVar("A")


class Timer:
    def __init__(self):
        self.tic = time.perf_counter()

    def __str__(self) -> str:
        toc = time.perf_counter()
        return str(datetime.timedelta(seconds=(toc - self.tic)))


def chunker(input_list, chunk_size=100):
    return [
        input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)
    ]


def multiprocess_function(
    processes: int, args: list[A], func: Callable[[A], T], chunksize: int = 1
) -> Iterator[T]:
    if processes == 0:
        for a in args:
            yield func(a)

    else:
        with multiprocessing.Pool(processes) as p:
            results = p.imap_unordered(func, args, chunksize=chunksize)

            for r in results:
                yield r
