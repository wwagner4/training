import functools as ft
from pathlib import Path
from typing import Callable

import simrunner as sr
import typer
from typing_extensions import Annotated

app = typer.Typer()

simpath_help = "Path to the simulator git module"
simpath_default = Path.home() / "prj" / "SUMOSIM" / "sumosim"


@app.command()
def sim(
    simulation_port: Annotated[
        int, typer.Option(help="The port on which the simulation is listening")
    ],
    simulation_path: Annotated[Path, typer.Option(help=simpath_help)] = simpath_default,
    verbose: Annotated[bool, typer.Option("-v", help="Verbose output")] = False,
):
    f = ft.partial(sr.run, simulation_port, simulation_path)
    _call(f, verbose)


@app.command()
def start(
    simulation_port: Annotated[
        int, typer.Option(help="The port on which the simulation is listening")
    ],
    base_port: Annotated[
        int,
        typer.Option(help="base port. E.g 455 will use ports 4550, 4551, 4552, ..."),
    ],
    verbose: Annotated[bool, typer.Option("-v", help="Verbose output")] = False,
):
    f = ft.partial(sr.start, base_port)
    _call(f, verbose)


def _call(f: Callable[[], None], verbose: bool):
    if verbose:
        f()
    else:
        try:
            f()
        except Exception as e:
            print(f"ERROR: {e}")


@app.command()
def tryout():
    print("tryout")


if __name__ == "__main__":
    app()
