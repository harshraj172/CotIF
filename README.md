# CotIF: Chain of Thought Instruction Following

## Installation

## Usage

### 1. Data Synthesis

Generate training datasets using the synthesis pipeline:

#### Standard Synthesis
```bash
python src/synthesize/main.py \
    --output-path /share/u/harshraj/CotIF/data/cotroller_train_dataset-mix-v2.json \
    --num-samples 10000
```

#### With AutoIF Enhancement
```bash
python src/synthesize/main.py \
    --output-path /share/u/harshraj/CotIF/data/cotroller_train_dataset-autoif-mix.json \
    --num-samples 10000 \
    --use-autoif
```

### 2. Data Visualization

Visualize and analyze generated samples using the web-based viewer:

```bash
cd src/viz
pip install flask
python chat_viewer.py
```

Then open your browser to `http://localhost:5000` and upload a JSONL file to view.

#### Example Dataset
An example dataset for visualization is available at:
```
data/for_viz/Bespoke-Stratos-17k-llama3-hehe-partial_soln-from_base.jsonl
```

This file contains samples generated using the annotation strategy described in the [paper](https://arxiv.org/abs/2504.05419v1).
