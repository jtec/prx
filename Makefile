test:
		uv run pytest -n auto --dist=loadfile --durations=10

format-lint:
		uv run ruff check --fix
		uv run ruff format

format-lint-check:
		uv run ruff check --exit-non-zero-on-fix
		uv run ruff format --check