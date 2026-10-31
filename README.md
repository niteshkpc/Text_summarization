# Text Summarization with T5

Fine-tune a T5 sequence-to-sequence model on the SAMSum dialogue summarization
dataset, then generate summaries from the command line. The notebook
[`Text_Summarizer.ipynb`](./Text_Summarizer.ipynb) contains the original
exploration; `train_summarizer.py` is the reproducible command-line workflow.

## Requirements

- Python 3.10–3.13
- pip
- A CUDA-capable GPU is recommended for training; CPU is supported but slower

Create and activate a virtual environment, then install dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Dataset

Download the SAMSum CSV files from the
[SAMSum dataset page](https://huggingface.co/datasets/Samsung/samsum) and place
`samsum-train.csv` and `samsum-validation.csv` in a `data/` directory. Each CSV
must contain `dialogue` and `summary` columns. The script also checks the
project root for these files.

Dataset files are intentionally excluded from Git. Check the dataset page for
its current license and terms before using or redistributing it.

## Train

Fine-tune on the full training and validation CSVs:

```powershell
python train_summarizer.py train
```

For a quick end-to-end smoke run, train on small samples:

```powershell
python train_summarizer.py train --max-train-samples 32 --max-eval-samples 16 --epochs 1
```

By default, the script starts from `t5-small`, writes training checkpoints to
`results/`, and saves the final model and tokenizer to `saved_summary_model/`.
If that model directory already contains a model config, subsequent training
continues from it. Both generated directories are excluded from Git.

Common options:

```text
--data-dir PATH              Dataset directory (default: data)
--base-model MODEL           Hugging Face model name (default: t5-small)
--model-dir PATH             Final model location (default: saved_summary_model)
--output-dir PATH            Trainer checkpoints location (default: results)
--epochs NUMBER              Training epochs (default: 3)
--batch-size NUMBER          Per-device batch size (default: 4)
--max-train-samples NUMBER   Optional training subset
--max-eval-samples NUMBER    Optional validation subset
```

## Summarize

Pass a dialogue as a quoted command-line argument:

```powershell
python train_summarizer.py summarize "Alex: Are we still meeting at 3? Sam: Yes, see you there."
```

The command uses `saved_summary_model/` if it contains a trained model.
Otherwise, it loads the base `t5-small` model. The first run downloads the
selected Hugging Face model; add `--base-model MODEL` to choose another one.
For long dialogues, use `--max-input-length` to adjust the input token limit.

## Project files

- `train_summarizer.py` — dataset preprocessing, training, and inference
- `Text_Summarizer.ipynb` — original Jupyter notebook
- `requirements.txt` — Python dependencies
