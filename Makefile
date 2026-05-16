test tests:
	uv run pytest -n auto --dist=loadfile --durations=10

format:
	uv run ruff format

format-check:
	uv run ruff format --check

type-check-fix:
	uv run ruff check --fix

type-check:
	uv run ruff check