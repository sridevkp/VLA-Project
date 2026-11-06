"""Command-line entry point for project tools."""

import typer

from vlarobot.backends.webots.smoke import create_webots_robot, run_smoke

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Project commands."""


@app.command()
def smoke() -> None:
    """Connect to Webots, read a camera frame, and advance ten simulation steps."""
    result = run_smoke(create_webots_robot())
    typer.echo(
        f"sim_time={result.sim_time:.3f}s "
        f"image_shape=({result.image_height}, {result.image_width}, 4)"
    )


if __name__ == "__main__":
    app()
