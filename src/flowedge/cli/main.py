"""FlowEdge CLI entry point."""

from typing import Annotated

import typer
from rich.console import Console

from flowedge.config.logging import setup_logging
from flowedge.ingestion.clone import validate_repo_url

app = typer.Typer(name="flowedge", help="FlowEdge Repo Intelligence Engine")
console = Console()


@app.command()
def analyze(
    repos: Annotated[
        list[str], typer.Argument(help="GitHub repo URLs to analyze")
    ],
    log_level: Annotated[str, typer.Option(help="Log level")] = "INFO",
) -> None:
    """Analyze one or more GitHub repos."""
    setup_logging(log_level)

    validated: list[str] = []
    for url in repos:
        try:
            validated.append(validate_repo_url(url))
        except ValueError as e:
            console.print(f"[red]Invalid URL:[/red] {e}")
            raise typer.Exit(1) from e

    console.print(f"[green]Analyzing {len(validated)} repo(s)...[/green]")
    for url in validated:
        console.print(f"  - {url}")

    # TODO: wire into LangGraph orchestration
    console.print("[yellow]Graph orchestration not yet wired.[/yellow]")


@app.command()
def version() -> None:
    """Show version."""
    console.print("flowedge 0.1.0")


if __name__ == "__main__":
    app()
