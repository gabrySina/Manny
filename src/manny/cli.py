"""Command line interface."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table

from manny import services
from manny.agents.spec import Action, AgentResult, Plan, TaskNode
from manny.bench import run_bench
from manny.checks import REPORT_NAME
from manny.config import Config, load_config
from manny.hardware import profiler
from manny.hardware.governor import MemoryGovernor, MemoryPressureError
from manny.llm.base import LLMBackendError
from manny.llm.catalog import ModelChoice, load_catalog, rank_choices
from manny.llm.ollama import MIN_STRUCTURED_VERSION, OllamaClient
from manny.runtime import SetupError, model_budget, run_task
from manny.sandbox.docker import SandboxError

app = typer.Typer(no_args_is_help=True, add_completion=False)
services_app = typer.Typer(no_args_is_help=True, help="Servizi locali di supporto (SearXNG per la ricerca web).")
app.add_typer(services_app, name="services")
console = Console()

# llama-server reads this to cap its in-RAM prompt cache, 8 GB by default.
CACHE_RAM_VARIABLE = "LLAMA_ARG_CACHE_RAM"
STATUS_STYLE = {"ok": "[green]ok[/green]", "warn": "[yellow]attenzione[/yellow]", "error": "[red]errore[/red]"}

Check = tuple[str, str, str]


@app.callback()
def main() -> None:
    """Manny: sistema multi-agente locale che si adatta all'hardware."""


def _docker_check() -> Check:
    if shutil.which("docker") is None:
        return "error", "Docker", "non installato: serve per eseguire e testare il codice"
    try:
        out = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return "error", "Docker", "il comando docker non risponde"
    if out.returncode != 0:
        return "error", "Docker", "demone non raggiungibile (è avviato? hai i permessi?)"
    return "ok", "Docker", out.stdout.strip()


def _searxng_check(cfg: Config) -> Check:
    if not services.searxng_healthy(cfg.search):
        return "warn", "SearXNG", "non attivo: avvialo con 'manny services up' per abilitare la ricerca web"
    return "ok", "SearXNG", cfg.search.searxng_url


def _model_check(choices: list[ModelChoice]) -> Check:
    if not choices:
        return "error", "Modello", "nessun modello del catalogo entra nel budget di memoria"
    best = choices[0]
    if not best.installed:
        return "warn", "Modello", f"da scaricare: ollama pull {best.name} (~{best.size_mb / 1024:.1f} GB)"
    return "ok", "Modello", best.name


@app.command()
def doctor(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="File di configurazione YAML."),
) -> None:
    """Rileva l'hardware, calcola il budget di memoria e verifica i prerequisiti."""
    cfg = load_config(config)
    profile = profiler.detect()
    governor = MemoryGovernor(cfg.memory, profile.ram_total_mb)
    client = OllamaClient(cfg.ollama.host)
    version = client.version()
    installed = client.installed() if version else set()
    budget = model_budget(governor, client) if version else governor.budget()
    choices = rank_choices(
        load_catalog(cfg.model.catalog_path),
        budget.ram_mb,
        budget.vram_mb,
        profile.disk_free_mb,
        installed,
        cfg.model,
        cfg.memory.disk_reserve_mb,
    )

    hw = Table(title="Hardware", show_header=False)
    hw.add_row("CPU", f"{profile.cpu_model} ({profile.physical_cores} core / {profile.logical_cores} thread)")
    hw.add_row("RAM", f"{profile.ram_total_mb} MB totali, {profile.ram_available_mb} MB disponibili")
    hw.add_row("Swap", f"{profile.swap_total_mb} MB")
    gpu = profile.gpu
    hw.add_row("GPU", f"{gpu.name}, {gpu.vram_total_mb} MB VRAM ({gpu.vram_free_mb} MB liberi)" if gpu else "nessuna GPU utilizzabile")
    hw.add_row("Disco modelli", f"{profile.disk_free_mb} MB liberi in {profile.models_dir}")
    hw.add_row("Sistema", f"{profile.os} {profile.arch}")
    hw.add_row("Fascia", profile.tier)
    console.print(hw)

    mem = Table(title="Budget di memoria", show_header=False)
    mem.add_row("Riserva RAM", f"{governor.ram_reserve_mb} MB")
    mem.add_row("RAM utilizzabile", f"{budget.ram_mb} MB")
    mem.add_row("VRAM utilizzabile", f"{budget.vram_mb} MB" if budget.vram_mb is not None else "-")
    console.print(mem)

    if choices:
        models = Table(title="Modelli adatti (il primo è quello scelto)")
        for column in ("Modello", "Contesto", "Dove", "Memoria stimata", "Installato"):
            models.add_column(column)
        for c in choices:
            models.add_row(c.name, str(c.num_ctx), c.placement, f"{c.est_mb} MB", "sì" if c.installed else "no")
        console.print(models)

    checks: list[Check] = []
    if version is None:
        checks.append(("error", "Ollama", f"non raggiungibile su {cfg.ollama.host}"))
    elif version < MIN_STRUCTURED_VERSION:
        found, wanted = (".".join(map(str, v)) for v in (version, MIN_STRUCTURED_VERSION))
        checks.append(("warn", "Ollama", f"versione {found}: aggiorna almeno alla {wanted} per gli output vincolati da schema"))
    else:
        checks.append(("ok", "Ollama", ".".join(map(str, version))))
    checks.append(_model_check(choices))
    checks.append(_docker_check())
    checks.append(_searxng_check(cfg))
    service_env = profiler.ollama_service_env()
    if service_env is not None and CACHE_RAM_VARIABLE not in service_env:
        checks.append((
            "warn", "Cache di Ollama",
            f"{CACHE_RAM_VARIABLE} non impostata nel servizio: la cache dei prompt può occupare fino a 8 GB di RAM",
        ))
    gpu_problem = profiler.gpu_problem()
    if gpu_problem:
        checks.append(("warn", "GPU", f"nvidia-smi non funziona ({gpu_problem}): la GPU viene ignorata; spesso basta riavviare"))
    if profile.has_avx2 is False:
        checks.append(("warn", "CPU", "senza AVX2: l'inferenza su CPU sarà molto lenta"))
    if profile.disk_free_mb < cfg.memory.disk_reserve_mb:
        checks.append(("warn", "Disco", "spazio libero sotto la riserva: non verranno scaricati nuovi modelli"))

    report = Table(title="Verifiche", show_header=False)
    for status, name, detail in checks:
        report.add_row(STATUS_STYLE[status], name, detail)
    console.print(report)

    if any(status == "error" for status, _, _ in checks):
        raise typer.Exit(code=1)


def _short(value: Any, limit: int = 80) -> str:
    text = str(value).replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "..."


def _print_event(source: str, kind: str, data: Any) -> None:
    if kind == "plan" and isinstance(data, Plan):
        console.print(f"Tipo di lavoro: [bold]{data.kind}[/bold]")
        for criterion in data.criteria:
            console.print(f"  criterio: {escape(criterion)}")
        team = Table(title="Squadra creata per questo task")
        for column in ("Agente", "Ruolo", "Strumenti"):
            team.add_column(column)
        for agent in data.agents:
            team.add_row(escape(agent.name), escape(agent.role), escape(", ".join(agent.tools)))
        console.print(team)
        work = Table(title="Piano di lavoro")
        for column in ("Id", "Agente", "Obiettivo", "Dipende da", "Test"):
            work.add_column(column)
        for node in data.tasks:
            work.add_row(
                escape(node.id), escape(node.agent), escape(node.goal),
                escape(", ".join(node.depends_on)), "sì" if node.gated else "",
            )
        console.print(work)
    elif kind == "task_start" and isinstance(data, TaskNode):
        console.rule(escape(f"{data.id}: {data.agent}"))
    elif kind == "action" and isinstance(data, Action):
        args = ", ".join(f"{key}={_short(value, 60)}" for key, value in data.args.items())
        console.print(f"[cyan]{escape(source)}[/cyan] {escape(data.tool)}({escape(args)})")
        if data.thought:
            console.print(f"  [dim]{escape(_short(data.thought, 200))}[/dim]")
    elif kind == "observation":
        console.print(f"  [dim]-> {escape(_short(data, 300))}[/dim]")
    elif kind == "final_check":
        passed, output = data
        status = "[green]superata[/green]" if passed else "[red]non superata[/red]"
        console.print(f"Verifica automatica: {status}")
        console.print(f"  [dim]{escape(_short(output, 600))}[/dim]")
    elif kind == "review":
        if not data:
            console.print("Revisione: [green]approvato[/green]")
        else:
            console.print(f"Revisione: [yellow]{len(data)} rilievi[/yellow]")
            for issue in data:
                console.print(f"  - {escape(issue)}")
    elif kind == "fix_result":
        issue, status = data
        label = {
            "fixed": "[green]risolto[/green]",
            "open": "[yellow]ancora aperto[/yellow]",
            "reverted": "[red]correzione annullata: rompeva le verifiche[/red]",
        }[status]
        console.print(f"Rilievo {label}: {escape(_short(issue, 160))}")
    elif kind == "task_end" and isinstance(data, AgentResult):
        status = "[green]completato[/green]" if data.success else "[red]fallito[/red]"
        console.print(f"{status} in {data.steps} passi: {escape(data.summary)}")


@app.command()
def run(
    task: str = typer.Argument(..., help="Il compito da svolgere, in linguaggio naturale."),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Cartella di lavoro (default: workspaces/<data-ora>)."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="File di configurazione YAML."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Usa questo modello Ollama invece di quello scelto in automatico."),
) -> None:
    """Svolge un task: crea gli agenti necessari, scrive il codice, lo testa e lo corregge."""
    cfg = load_config(config)
    workspace = workspace or Path("workspaces") / datetime.now().strftime("%Y%m%d-%H%M%S")
    console.print(f"Cartella di lavoro: {workspace.resolve()}")
    try:
        report = run_task(task, workspace, cfg, _print_event, model)
    except (SetupError, SandboxError, MemoryPressureError, LLMBackendError) as exc:
        console.print(f"[red]Errore:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    console.rule("Risultato")
    summary = Table(show_header=False)
    if not report.result.success:
        outcome = "[red]non riuscito[/red]"
    elif report.result.issues:
        outcome = f"[yellow]riuscito, con {len(report.result.issues)} rilievi di revisione aperti[/yellow]"
    else:
        outcome = "[green]riuscito[/green]"
    summary.add_row("Esito", outcome)
    summary.add_row("Modello", f"{report.model.name} (contesto {report.model.num_ctx}, {report.model.placement})")
    if report.tokens_per_second:
        summary.add_row("Velocità", f"{report.tokens_per_second:.0f} token/s")
    summary.add_row("Ricerca web", "attiva" if report.web_enabled else "non attiva")
    summary.add_row("RAM disponibile minima", f"{report.peak.min_ram_available_mb} MB")
    if report.peak.min_vram_free_mb is not None:
        summary.add_row("VRAM libera minima", f"{report.peak.min_vram_free_mb} MB")
    summary.add_row("Crescita dello swap", f"{report.peak.swap_growth_mb} MB")
    console.print(summary)
    if report.result.restored:
        console.print("La cartella funzionava prima di questa sessione: è stata riportata allo stato iniziale.")
    for issue in report.result.issues:
        console.print(f"  rilievo aperto: {escape(issue)}")
    answer = workspace / REPORT_NAME
    if report.result.kind == "research" and answer.is_file():
        console.rule(REPORT_NAME)
        console.print(Markdown(answer.read_text(encoding="utf-8")))
    if not report.result.success:
        raise typer.Exit(code=1)


@app.command()
def bench(
    models: list[str] = typer.Argument(..., help="Modelli Ollama da provare; aggiungi @think o @nothink per forzare il ragionamento."),
    tasks: Optional[list[str]] = typer.Option(None, "--task", "-t", help="Limita ai compiti indicati."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Cartella dei risultati (default: bench-runs/<data-ora>)."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="File di configurazione YAML."),
) -> None:
    """Misura qualità, velocità e memoria di uno o più modelli su un set fisso di compiti."""
    cfg = load_config(config)
    out = out or Path("bench-runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        outcomes = run_bench(models, cfg, out, tasks, lambda line: console.print(escape(line)))
    except ValueError as exc:
        console.print(f"[red]Errore:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    (out / "results.json").write_text(json.dumps([o.as_dict() for o in outcomes], indent=2), encoding="utf-8")

    table = Table(title="Risultati del banco di prova")
    for column in ("Modello", "Punteggio", "Compiti", "Tempo", "Token/s", "Dove", "VRAM", "RAM del modello", "RAM libera minima"):
        table.add_column(column)
    for o in sorted(outcomes, key=lambda o: -o.score):
        detail = " ".join(f"{t.task}:{t.passed}/{t.total}" for t in o.tasks)
        table.add_row(
            escape(o.spec), f"{o.score:.0f}", escape(detail), f"{o.seconds:.0f}s", f"{o.tokens_per_second:.0f}",
            f"{o.placement} ({o.num_ctx})", f"{o.model_vram_mb} MB", f"{o.runner_ram_mb} MB", f"{o.min_ram_available_mb} MB",
        )
    console.print(table)
    console.print(f"Dettagli in {out.resolve()}")


@services_app.command("up")
def services_up(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Avvia SearXNG in un container, raggiungibile solo da questo computer."""
    cfg = load_config(config)
    try:
        services.searxng_up(cfg.search)
    except services.ServiceError as exc:
        console.print(f"[red]Errore:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    console.print(f"SearXNG attivo su {cfg.search.searxng_url}")


@services_app.command("down")
def services_down() -> None:
    """Ferma e rimuove il container di SearXNG."""
    services.searxng_down()
    console.print("SearXNG fermato")


@services_app.command("status")
def services_status(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Mostra se SearXNG risponde."""
    cfg = load_config(config)
    console.print("SearXNG attivo" if services.searxng_healthy(cfg.search) else "SearXNG non attivo")
