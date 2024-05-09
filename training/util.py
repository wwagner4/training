import functools as ft
from concurrent.futures import ThreadPoolExecutor


def message(ex: BaseException) -> str:
    msg = str(ex)
    if msg:
        return msg
    return str(type(ex))


def run_concurrent(functions: list, max_workers: int) -> list:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        partials = [ft.partial(*function) for function in functions]
        futures = [executor.submit(p) for p in partials]
        return [future.result() for future in futures]
