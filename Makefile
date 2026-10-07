.PHONY: setup download-data train train-dscnn evaluate export quantize benchmark demo clean all

setup:
	uv venv
	uv pip install -r requirements.txt

download-data:
	bash scripts/download_data.sh

train:
	uv run python src/train.py --config configs/default.yaml

train-dscnn:
	uv run python src/train.py --config configs/dscnn.yaml

evaluate:
	uv run python src/evaluate.py --config configs/default.yaml

export:
	uv run python src/export_onnx.py --config configs/default.yaml

quantize:
	uv run python src/quantize.py --config configs/default.yaml

benchmark:
	uv run python src/benchmark.py --config configs/default.yaml

demo:
	cd web_demo && uv run python -m http.server 8000

clean:
	rm -rf .venv
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf data/speech_commands_v0.02.tar.gz

all:
	bash scripts/run_full_pipeline.sh
