"""Command-line entry point for project tools."""

import typer

from vlarobot.backends.webots.smoke import create_webots_robot, run_smoke
from vlarobot.backends.zmq.client import ZmqBackendClient
from vlarobot.backends.zmq.server import DEFAULT_ENDPOINT
from vlarobot.core.types import ScenarioSpec
from vlarobot.execution.scripted_can import ScriptedCanPolicy

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


def run_webots_sim_server(endpoint: str = DEFAULT_ENDPOINT) -> None:
    """Run the Webots-side server; also used by the Webots controller shim."""
    from vlarobot.backends.webots.server import run_webots_sim_server as run_server

    run_server(endpoint)


@app.command("sim-server")
def sim_server(
    endpoint: str = typer.Option(DEFAULT_ENDPOINT, help="ZeroMQ REP bind endpoint."),
) -> None:
    """Serve lockstep Webots requests; run from the Webots extern controller."""
    run_webots_sim_server(endpoint)


@app.command()
def brain(
    endpoint: str = typer.Option(DEFAULT_ENDPOINT, help="ZeroMQ simulator endpoint."),
    steps: int = typer.Option(10, min=1, help="Number of lockstep simulation steps."),
) -> None:
    """Connect to the simulator, reset the default scenario, and step the brain-side client."""
    client = ZmqBackendClient(endpoint)
    try:
        client.ping()
        observation = client.reset(ScenarioSpec(), seed=0)
        for _ in range(steps):
            observation = client.step()
        typer.echo(f"sim_time={observation.t:.3f}s image_shape={observation.rgb.shape}")
    finally:
        client.close()


@app.command("can-regression")
def can_regression(
    endpoint: str = typer.Option(DEFAULT_ENDPOINT, help="ZeroMQ simulator endpoint."),
    seeds: int = typer.Option(10, min=1, help="Number of seeded CAN episodes to run."),
) -> None:
    """Run a brain-side scripted CAN policy over repeated seeds and summarize the outcome."""
    client = ZmqBackendClient(endpoint)
    policy = ScriptedCanPolicy()
    results = []
    try:
        client.ping()
        for seed in range(seeds):
            result = policy.execute(client, seed)
            results.append(result)
            typer.echo(
                f"seed={seed} success={result.success} rejections={result.rejections} "
                f"final_distance={result.final_target_distance:.3f}m state={result.final_state}"
            )
        successes = sum(1 for result in results if result.success)
        rejections = sum(result.rejections for result in results)
        typer.echo(f"summary success={successes}/{seeds} rejections={rejections}")
    finally:
        client.close()


if __name__ == "__main__":
    app()
