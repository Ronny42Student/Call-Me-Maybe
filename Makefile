SGOINFRE := $(shell if [ -d "$(HOME)/sgoinfre" ]; then echo "$(HOME)/sgoinfre"; elif [ -d "/sgoinfre/$(USER)" ]; then echo "/sgoinfre/$(USER)"; fi)

ifneq ($(SGOINFRE),)
    export UV_CACHE_DIR := $(SGOINFRE)/uv_cache
    export HF_HOME := $(SGOINFRE)/hf_cache
endif

.PHONY: install run debug clean lint lint-strict

install:
	@if [ -n "$(UV_CACHE_DIR)" ]; then \
		mkdir -p $(UV_CACHE_DIR); \
		if [ ! -L "$(HOME)/.cache/uv" ]; then \
			rm -rf "$(HOME)/.cache/uv"; \
			ln -s "$(UV_CACHE_DIR)" "$(HOME)/.cache/uv"; \
		fi; \
	fi
	uv sync
	uv pip install flake8

run:
	uv run python -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calls.json

debug:
	uv run python -m pdb -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calling_results.json

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "output" -exec rm -rf {} +

lint:
	uv run --with flake8 --with mypy src/* --exclude=.venv
	uv run --with mypy mypy src/* --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run --with flake8 flake8 src/* --exclude=.venv
	uv run --with mypy mypy src/* --strict
