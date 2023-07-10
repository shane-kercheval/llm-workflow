.PHONY: tests

####
# docker commands
####
docker_build:
	# build the docker container used to run tests and build package
	docker compose -f docker-compose.yml build

docker_run: docker_build
	# run the docker container
	docker compose -f docker-compose.yml up

docker_down:
	docker compose down --remove-orphans

docker_rebuild:
	# rebuild docker container
	docker compose -f docker-compose.yml build --no-cache

docker_zsh:
	# run container and open up zsh command-line
	docker exec -it python-helpers-bash-1 /bin/zsh

####
# project commands
####
# commands to run inside docker container

linting:
	ruff check llm_workflow/
	ruff check tests/

unittests:
	rm -f tests/test_files/log.log
	# pytest tests
	coverage run -m pytest --durations=0 tests
	coverage html

doctests:
	python -m doctest llm_workflow/utilities.py

tests: linting unittests doctests

docker_tests:
	# run tests within docker container
	docker compose run --no-deps --entrypoint "make tests" bash

open_coverage:
	open 'htmlcov/index.html'

####
# Package Build
####

## Build package
package: clean
	# NOTE: make sure .pypirc file is in home directory
	# cp .pypirc ~/
	rm -fr dist
	python -m build
	twine upload dist/*

docker_package:
	# create package and upload via twine from within docker container
	docker compose run --no-deps --entrypoint "make package" bash

## Delete all generated files
clean:
	rm -rf dist
	rm -rf llm_workflow.egg-info
