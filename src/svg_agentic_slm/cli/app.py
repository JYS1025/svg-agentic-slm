"""Main Typer application and command registration."""

import typer

from svg_agentic_slm.cli.commands_generate import generate
from svg_agentic_slm.cli.commands_validate import validate
from svg_agentic_slm.cli.commands_render import render
from svg_agentic_slm.cli.commands_train import train
from svg_agentic_slm.cli.commands_eval import evaluate

app = typer.Typer(
    name="svg-agentic-slm",
    help="Agentic SLM-based SVG generation pipeline.",
    add_completion=False,
)

app.command(name="generate")(generate)
app.command(name="validate")(validate)
app.command(name="render")(render)
app.command(name="train")(train)
app.command(name="eval")(evaluate)


if __name__ == "__main__":
    app()
