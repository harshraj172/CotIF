# CotIF
Chain of Though Instruction Following

## Data Synthesis

```bash
python src/synthesize/main.py --output-path /share/u/harshraj/CotIF/data/cotroller_train_dataset-mix-v2.js
on --num-samples 10000
```


**With AutoIf**

```bash
python src/synthesize/main.py --output-path /share/u/harshraj/CotIF/data/cotroller_train_dataset-autoif-mix.js
on --num-samples 10000 --use-autoif
```