import os

from invoke import Context, task

WINDOWS = os.name == "nt"


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run pytest tests/", echo=True, pty=not WINDOWS)


@task
def data(ctx: Context, subset: str = "small", limit: int = 0) -> None:
    """Download FMA and process it (limit=0 processes every track)."""
    limit_flag = f" --limit {limit}" if limit else ""
    ctx.run(f"uv run --extra pipeline python -m afp.data --subset {subset}{limit_flag}", echo=True, pty=not WINDOWS)


@task
def fmt(ctx: Context) -> None:
    """Auto-fix lint issues and format the code."""
    ctx.run("uv run ruff check . --fix", echo=True, pty=not WINDOWS)
    ctx.run("uv run ruff format .", echo=True, pty=not WINDOWS)
