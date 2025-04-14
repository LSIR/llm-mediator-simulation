"""Local server to run a model locally and keep it loaded for faster prompting.

Usage guide:

## Run the server in the foreground (debug)

```bash
python examples/example_server.py server
```

## Run the server in the background

```bash
python examples/example_server.py start
```

You can test if the server is available by running the following command:

```bash
curl localhost:8000
watch -n 1 curl localhost:8000  # To check every second
```

The server may take a few seconds to start because it must load the model first.

## Stop the server

```bash
python examples/example_server.py stop
```

## Call the server

### With an inline prompt

```bash
python examples/example_server.py call -p "Hello, how are you ?"
```

### With a prompt from a file

```bash
python examples/example_server.py call -f prompt.txt
```
"""

import click
import httpx

PORT = 8000


@click.command("start")
@click.option(
    "--model_name",
    "-m",
    default="/mnt/datastore/models/mistralai/Mistral-7B-Instruct-v0.2",
    help="The model name to load.",
)
def start(model_name="/mnt/datastore/models/mistralai/Mistral-7B-Instruct-v0.2"):
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
                "scripts.hf_server",
                "server",
                "--model_name",
                model_name,
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
            json={"text": prompt, "seed": None},
            timeout=40,
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
@click.option(
    "--model_name",
    "-m",
    default="/mnt/datastore/models/mistralai/Mistral-7B-Instruct-v0.2",
    help="The model name to load.",
)
def server(model_name="/mnt/datastore/models/mistralai/Mistral-7B-Instruct-v0.2"):
    """Start a Flask server to keep the LLM loaded"""
    from llm_mediator_simulation.models.hf_local_model import HFLocalModel

    # Load the model

    model = HFLocalModel(
        model_name=model_name,
        max_new_tokens=500,
        json=True,
    )
    from flask import Flask, request

    app = Flask("LLM Server")

    @app.route("/")
    def home():  # type: ignore
        return "Local LLM Server"

    @app.route("/call", methods=["POST"])
    def call():  # type: ignore
        data = request.get_json()
        text = data.get("text")
        seed = data.get("seed")

        return model.sample(text, seed=seed)

    @app.route("/stop")
    def stop():  # type: ignore
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
