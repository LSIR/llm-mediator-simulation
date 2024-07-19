"""Local server to run a model locally and keep it loaded for faster prompting."""

import click
import httpx

PORT = 8000


@click.command("start")
def start():
    """Launches the local server"""
    import subprocess

    # Test is the server is already running
    try:
        httpx.get(f"http://localhost:{PORT}")
        click.echo("Local server already running.")
        return
    except httpx.ConnectError:
        pass

    # Launch the server and write the logs to a logfile
    with open("process.log", "a") as log_file:
        process = subprocess.Popen(
            [
                "python",
                "-m",
                "examples.example_server",
                "server",
            ],
            stdout=log_file,
            stderr=log_file,
            universal_newlines=True,
        )

    click.echo(f"Model started (PID {process.pid}).")


@click.command("stop")
def stop():
    """Stops the local server"""

    try:
        httpx.get(f"http://localhost:{PORT}/stop")
    except httpx.ConnectError:
        click.echo("Local server not running.")
        return
    except httpx.RemoteProtocolError:
        click.echo("Local server stopped.")
        return

    click.echo("Something went wrong.")


@click.command("call")
@click.option("--prompt", "-p", help="The prompt to send to the local server.")
@click.option(
    "--file",
    "-f",
    help="The file from which to read the prompt to send to the local server.",
)
def call(prompt: str | None, file: str | None):
    """Calls the local server"""

    if prompt is None:
        if file is None:
            click.echo("Please provide a prompt or a file.")
            return
        with open(file, "r") as f:
            prompt = f.read()

    try:
        response = httpx.post(
            f"http://localhost:{PORT}/call",
            json={"text": prompt},
        )
        click.echo(response.text)
    except httpx.ConnectError:
        click.echo("Local server not running.")


@click.group()
def main():
    pass


###################################################################################################
#                                             LLM SERVER                                          #
###################################################################################################


@click.command("server")
def server():
    """Start a Flask server to keep the LLM loaded"""
    from llm_mediator_simulation.models.mistral_local_model import MistralLocalModel

    # Load the model
    model = MistralLocalModel(max_length=500)

    from flask import Flask, request

    app = Flask("LLM Server")

    @app.route("/")
    def home():
        return "Local LLM Server"

    @app.route("/call", methods=["POST"])
    def call():
        data = request.get_json()
        text = data.get("text")

        return model.sample(text)

    @app.route("/stop")
    def stop():
        """Stop the server"""
        import os

        os._exit(0)

    app.run(port=PORT)


main.add_command(start)
main.add_command(stop)
main.add_command(call)
main.add_command(server)

if __name__ == "__main__":
    main()
