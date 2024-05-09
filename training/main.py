import functools as ft
from pathlib import Path
from typing import Callable

import typer
from typing_extensions import Annotated

import training.simdb
import training.simrunner as sr
import training.tryout as to
import training.util

app = typer.Typer()

simpath_help = "Path to the simulator git module"
simpath_default = Path.home() / "prj" / "SUMOSIM" / "sumosim"


@app.command()
def start(
    port: Annotated[
        int, typer.Option(help="The port on which the simulation is listening")
    ],
    verbose: Annotated[bool, typer.Option("-v", help="Verbose output")] = False,
):
    f = ft.partial(sr.start, port)
    _call(f, verbose)


@app.command()
def tryout(
    verbose: Annotated[bool, typer.Option("-v", help="Verbose output")] = False,
):
    _call(to.main, verbose)


@app.command()
def db(
    query: Annotated[
        str,
        typer.Option(
            help="Name of a query function in module 'simdb'. E.g. 'count_running'"
        ),
    ],
    verbose: Annotated[bool, typer.Option("-v", help="Verbose output")] = False,
):
    callable = getattr(training.simdb, query)
    _call(callable, verbose)


def _call(f: Callable[[], None], verbose: bool):
    if verbose:
        f()
    else:
        try:
            f()
        except Exception as e:
            msg = training.util.message(e)
            print(f"ERROR: {msg}")


if __name__ == "__main__":
    app()
