"""Fine-tune a T5 model on SAMSum and summarize dialogues."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_MODEL = "t5-small"
TRAIN_FILE = "samsum-train.csv"
VALIDATION_FILE = "samsum-validation.csv"


def clean_text(text: str) -> str:
    """Normalize whitespace and remove markup from a dialogue or summary."""
    text = re.sub(r"\r\n", " ", str(text))
    text = re.sub(r"<.*?>", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def find_data_file(data_dir: Path, filename: str) -> Path:
    """Find a dataset file in the requested directory or the project root."""
    candidates = (data_dir / filename, Path(filename))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find {filename}. Place it in --data-dir or the project root "
        f"(checked: {expected})."
    )


class SummarizationDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: object,
        max_input_length: int,
        max_target_length: int,
    ) -> None:
        required_columns = {"dialogue", "summary"}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(
                f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}"
            )

        dialogues = frame["dialogue"].map(clean_text).tolist()
        summaries = frame["summary"].map(clean_text).tolist()
        self.encodings = tokenizer(
            dialogues,
            text_target=summaries,
            max_length=max_input_length,
            max_target_length=max_target_length,
            truncation=True,
        )

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict:
        return {key: values[index] for key, values in self.encodings.items()}


def select_device() -> torch.device:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_name(model_dir: Path, fallback_model: str) -> str:
    if (model_dir / "config.json").is_file():
        return str(model_dir)
    return fallback_model


def train(args: argparse.Namespace) -> None:
    import pandas as pd
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    train_path = find_data_file(args.data_dir, TRAIN_FILE)
    validation_path = find_data_file(args.data_dir, VALIDATION_FILE)
    train_frame = pd.read_csv(train_path)
    validation_frame = pd.read_csv(validation_path)

    if args.max_train_samples is not None:
        train_frame = train_frame.sample(
            n=min(args.max_train_samples, len(train_frame)), random_state=args.seed
        )
    if args.max_eval_samples is not None:
        validation_frame = validation_frame.sample(
            n=min(args.max_eval_samples, len(validation_frame)), random_state=args.seed
        )

    model_name = load_model_name(args.model_dir, args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(model_name, extra_special_tokens={})
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_dataset = SummarizationDataset(
        train_frame, tokenizer, args.max_input_length, args.max_target_length
    )
    validation_dataset = SummarizationDataset(
        validation_frame, tokenizer, args.max_input_length, args.max_target_length
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
    )

    trainer.train()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.model_dir))
    tokenizer.save_pretrained(args.model_dir)
    print(f"Fine-tuned model saved to {args.model_dir}")


def summarize(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = load_model_name(args.model_dir, args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(model_name, extra_special_tokens={})
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = select_device()
    model.to(device)
    model.eval()

    inputs = tokenizer(
        clean_text(args.dialogue),
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_length,
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune T5 on SAMSum or summarize a dialogue."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Fine-tune a model on SAMSum CSVs.")
    train_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    train_parser.add_argument("--base-model", default=DEFAULT_MODEL)
    train_parser.add_argument("--model-dir", type=Path, default=Path("saved_summary_model"))
    train_parser.add_argument("--output-dir", type=Path, default=Path("results"))
    train_parser.add_argument("--epochs", type=float, default=3)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--max-input-length", type=int, default=512)
    train_parser.add_argument("--max-target-length", type=int, default=150)
    train_parser.add_argument("--max-train-samples", type=int)
    train_parser.add_argument("--max-eval-samples", type=int)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train)

    summarize_parser = commands.add_parser(
        "summarize", help="Generate a summary for a dialogue."
    )
    summarize_parser.add_argument("dialogue", help="Dialogue text to summarize.")
    summarize_parser.add_argument(
        "--model-dir", type=Path, default=Path("saved_summary_model")
    )
    summarize_parser.add_argument("--base-model", default=DEFAULT_MODEL)
    summarize_parser.add_argument("--max-input-length", type=int, default=512)
    summarize_parser.add_argument("--max-new-tokens", type=int, default=150)
    summarize_parser.add_argument("--num-beams", type=int, default=4)
    summarize_parser.set_defaults(func=summarize)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "summarize" and not args.dialogue.strip():
        raise SystemExit("Dialogue text cannot be empty.")
    args.func(args)


if __name__ == "__main__":
    main()
