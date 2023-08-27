.PHONY: clean data lint requirements 

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = supg

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_environment:
	conda env create --file environment.yml

env_to_kernel:
	python3 -m ipykernel install --user --name $(PROJECT_NAME) --display-name $(PROJECT_NAME)

