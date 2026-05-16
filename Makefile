test tests:
	uv run pytest -n auto --dist=loadfile --durations=10

format-lint:
	uv run ruff format
	uv run ruff check --fix

format-lint-check:
	uv run ruff format --check
	uv run ruff check