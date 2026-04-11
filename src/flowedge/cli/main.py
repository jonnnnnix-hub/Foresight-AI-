"""FlowEdge CLI entry point."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.table import Table

from flowedge.config.logging import setup_logging
from flowedge.graph.pipeline import compile_analysis_graph
from flowedge.graph.state import GraphState
from flowedge.ingestion.clone import validate_repo_url
from flowedge.synthesis.export import export_json, export_markdown

app = typer.Typer(name="flowedge", help="FlowEdge Repo Intelligence Engine")
console = Console()


@app.command()
def analyze(
    repos: Annotated[list[str], typer.Argument(help="GitHub repo URLs to analyze")],
    log_level: Annotated[str, typer.Option(help="Log level")] = "INFO",
    output_dir: Annotated[str, typer.Option(help="Output directory for reports")] = "./output",
) -> None:
    """Analyze one or more GitHub repos and produce a comparison report."""
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

    # Run the analysis pipeline
    state = GraphState(repo_urls=validated)
    graph = compile_analysis_graph()
    final_state = asyncio.run(_run_pipeline(graph, state))

    if final_state.errors:
        console.print("\n[yellow]Warnings/Errors:[/yellow]")
        for err in final_state.errors:
            console.print(f"  [red]- {err}[/red]")

    if final_state.report:
        report = final_state.report
        out = Path(output_dir)

        # Export reports
        json_path = export_json(report, out / f"{report.run_id}.json")
        md_path = export_markdown(report, out / f"{report.run_id}.md")
        console.print("\n[green]Reports saved:[/green]")
        console.print(f"  JSON: {json_path}")
        console.print(f"  Markdown: {md_path}")

        # Print ranking table
        if report.rankings:
            table = Table(title="Repo Rankings")
            table.add_column("Rank", justify="center")
            table.add_column("Repository")
            table.add_column("Score", justify="right")
            table.add_column("Best For")
            table.add_column("Weakest At")
            for entry in report.rankings:
                table.add_row(
                    str(entry.rank),
                    entry.repo_name,
                    str(entry.weighted_score),
                    ", ".join(entry.best_for),
                    ", ".join(entry.weakest_at),
                )
            console.print(table)

        console.print(f"\n[bold]{report.executive_summary}[/bold]")
    else:
        console.print("[red]Pipeline failed to produce a report.[/red]")
        raise typer.Exit(1)


async def _run_pipeline(
    graph: CompiledStateGraph, state: GraphState  # type: ignore[type-arg]
) -> GraphState:
    """Execute the compiled graph pipeline."""
    result = await graph.ainvoke(state)
    if isinstance(result, dict):
        return GraphState.model_validate(result)
    return GraphState.model_validate(result)


@app.command()
def version() -> None:
    """Show version."""
    console.print("flowedge 0.1.0")


if __name__ == "__main__":
    app()
