run-docker-compose:
	uv sync
	docker compose up --build

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb

run-evals-retriever:
	uv sync
	uv run --env-file .env python -c "import sys; sys.path.extend(['apps/api','apps/api/src']); exec(open('apps/api/evals/eval_retriever.py').read())"