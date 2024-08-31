from pathlib import Path

import typer
from typing_extensions import Annotated

import training.simdb
import training.simrunner as sr
import training.tryout as to
import training.util

app = typer.Typer(pretty_exceptions_enable=False)

simpath_help = "Path to the simulator git module"
simpath_default = Path.home() / "prj" / "SUMOSIM" / "sumosim"


@app.command()
def start(
    sim_name: Annotated[str, typer.Option(help="Simulation name")],
    port: Annotated[
        int, typer.Option(help="The port on which the simulation is listening")
    ] = 4444,
    controller1: Annotated[
        sr.ControllerName, typer.Option(help="Name of controller 1")
    ] = sr.ControllerName.TUMBLR,
    controller2: Annotated[
        sr.ControllerName, typer.Option(help="Name of controller 1")
    ] = sr.ControllerName.STAY_IN_FIELD,
):
    sr.start(port, sim_name, controller1, controller2)


@app.command()
def tryout():
    to.main()


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
    training.simdb(query)


if __name__ == "__main__":
    app()
