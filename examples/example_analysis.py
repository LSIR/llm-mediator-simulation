"""An example analysis script run via CLI to analyze a pickled debate simulation."""

import click


def common_options(func):
    """Common options for the analysis commands."""
    func = click.option(
        "--debate", "-d", help="The pickled debate to analyze.", required=True
    )(func)
    func = click.option(
        "--average", "-a", help="Average the values among debaters.", is_flag=True
    )(func)
    return func


@click.command("metrics")
@common_options
def metrics(debate: str, average: bool):
    """Plot the debater metrics"""

    print(debate, average)


@click.command("personalities")
@common_options
def personalities(debate: str, average: bool):
    """Plot the debater personalities"""

    print(debate, average)


@click.group()
def main():
    pass


main.add_command(metrics)
main.add_command(personalities)


if __name__ == "__main__":
    main()
