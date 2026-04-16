.PHONY: install install-runtime install-demo install-dev train eval demo webui gguf chat test release-check clean

install:
	pip install -e .

install-runtime:
	pip install -e ".[runtime]"

install-demo:
	pip install -e ".[demo]"

install-dev:
	pip install -e ".[dev]"

train:
	python scripts/train_sft_dpo.py

eval:
	python evaluation/benchmark.py

demo:
	python app.py

webui:
	python app.py

gguf:
	bash scripts/convert_gguf.sh

chat:
	python -m aksarallm.inference

test:
	python -m unittest discover -s tests

release-check:
	python scripts/release_check.py

clean:
	rm -rf build dist *.egg-info __pycache__
