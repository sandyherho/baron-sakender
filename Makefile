# Makefile for baron-sakender

.PHONY: install dev test clean run-all run-case1 run-case2 run-case3 run-case4 help

help:
	@echo "baron-sakender - 2D Ideal MHD Solver"
	@echo ""
	@echo "Installation:"
	@echo "  make install     - Install package with pip"
	@echo "  make dev         - Install in development mode with poetry"
	@echo ""
	@echo "Testing:"
	@echo "  make test        - Run all tests"
	@echo "  make test-cov    - Run tests with coverage"
	@echo ""
	@echo "Running simulations:"
	@echo "  make run-all     - Run all test cases"
	@echo "  make run-case1   - Orszag-Tang vortex"
	@echo "  make run-case2   - Strong magnetic field"
	@echo "  make run-case3   - Current sheet"
	@echo "  make run-case4   - Alfven wave"
	@echo ""
	@echo "GPU acceleration:"
	@echo "  make run-case1-gpu - Run case1 with GPU"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean       - Remove outputs and cache"

install:
	pip install .

dev:
	poetry install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/baron_sakender --cov-report=html

run-all:
	baron-sakender --all

run-case1:
	baron-sakender case1

run-case2:
	baron-sakender case2

run-case3:
	baron-sakender case3

run-case4:
	baron-sakender case4

run-case1-gpu:
	baron-sakender case1 --gpu

run-case2-gpu:
	baron-sakender case2 --gpu

lint:
	black src/ tests/ --check

format:
	black src/ tests/

clean:
	rm -rf outputs/
	rm -rf logs/
	rm -rf __pycache__/
	rm -rf src/baron_sakender/__pycache__/
	rm -rf src/baron_sakender/core/__pycache__/
	rm -rf src/baron_sakender/io/__pycache__/
	rm -rf src/baron_sakender/utils/__pycache__/
	rm -rf src/baron_sakender/visualization/__pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf dist/
	rm -rf build/
	rm -rf htmlcov/
	rm -rf .coverage
