.PHONY: install init seed serve test eval grid prompt docker-up docker-down clean

install:
	python -m pip install -e ".[dev]"

init:
	python -m app.cli init-db

seed:
	python -m app.cli seed

serve:
	python -m app.cli serve

test:
	pytest -q

eval:
	python -m app.cli eval --kind full --limit 25 --top-k 6

grid:
	python -m app.cli experiments --kind retrieval_grid --limit 25

prompt:
	python -m app.cli experiments --kind prompt_grid --limit 10

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
