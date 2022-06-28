import datetime
import time


class Timer:
    def __init__(self):
        self.tic = time.perf_counter()

    def __str__(self) -> str:
        toc = time.perf_counter()
        return str(datetime.timedelta(seconds=(toc - self.tic)))
