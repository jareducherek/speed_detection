.PHONY: requirements

PROJECT_NAME=speed_detection

create_environment:
	conda create --yes --name $(PROJECT_NAME) --clone base

requirements:
	pip install -r requirements.txt
	pip install git+https://github.com/CSAILVision/semantic-segmentation-pytorch.git@master
	python -m ipykernel install --user
	python -m ipykernel install --user --name $(PROJECT_NAME) --display-name "$(PROJECT_NAME)"
	conda install cudatoolkit=10.2
	cd source/segmentation_config/ && \
	chmod +x get_weights.sh && \
	./get_weights.sh
