"""QLoRA LFM3 : réponses assistant seules, contrôle numérique et reprise de checkpoints."""
import collections
import gc
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainerCallback, TrainingArguments, set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# Une réexécution libère les références à l'ancienne tentative avant de recharger.
for old_name in ("trainer", "model", "train_result"):
    globals().pop(old_name, None)
gc.collect()
torch.cuda.empty_cache()

WORK_DIR = Path(globals().get("WORK_DIR", "/content/lfm3_cyber_ft"))
SAVE_DIR = Path(globals().get("SAVE_DIR", "/content/drive/MyDrive/LFM3_Kali_Finetune/run_20261105_v1"))
MODEL_ID = globals().get("MODEL_ID", "OpenIntelligenceNet/LFM3-1.7B-Uncensored")
MODEL_REVISION = globals().get("MODEL_REVISION", "8e7cf5cccd95afa9d34a3a66ebfb8fef1b55c204")
MAX_LENGTH = int(globals().get("MAX_LENGTH", 1536))
LORA_RANK = int(globals().get("LORA_RANK", 16))
LEARNING_RATE = float(globals().get("LEARNING_RATE", 1e-4))
GRADIENT_ACCUMULATION = int(globals().get("GRADIENT_ACCUMULATION", 8))
EPOCHS = int(globals().get("EPOCHS", 1))
USE_4BIT = bool(globals().get("USE_4BIT", True))
DATA_DIR = WORK_DIR / "data"
CHECKPOINT_DIR = SAVE_DIR / "checkpoints"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
set_seed(42)

if not torch.cuda.is_available():
    raise RuntimeError("L'entraînement nécessite le GPU T4.")
if not Path("/content/drive/MyDrive").is_dir():
    raise RuntimeError("Monter Google Drive pour sauvegarder et reprendre l'entraînement.")

print("Chargement du tokenizer…", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, token=False, padding_side="right"
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def encode_examples(records):
    accepted, skipped = [], collections.Counter()
    for row in records:
        encoded = tokenizer.apply_chat_template(
            row["messages"], tools=row.get("tools") or None,
            tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
            enable_thinking=False,
        )
        ids = encoded["input_ids"]
        mask = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))
        if not mask or not any(mask) or len(mask) != len(ids):
            raise RuntimeError("Le template ne fournit pas un masque assistant valide.")
        if len(ids) > MAX_LENGTH:
            skipped["too_long_no_truncation"] += 1
            continue
        labels = [token if yes else -100 for token, yes in zip(ids, mask)]
        accepted.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    return accepted, dict(skipped)


print("Tokenisation au format natif LFM3 ; seuls les tokens assistant sont supervisés…", flush=True)
train_rows = read_jsonl(DATA_DIR / "train.jsonl")
validation_rows = read_jsonl(DATA_DIR / "validation.jsonl")
train_encoded, train_skipped = encode_examples(train_rows)
valid_encoded, valid_skipped = encode_examples(validation_rows)
if not train_encoded or not valid_encoded:
    raise RuntimeError("Corpus tokenisé vide.")


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def collate(examples):
    length = math.ceil(max(len(row["input_ids"]) for row in examples) / 8) * 8
    return {
        "input_ids": torch.tensor([x["input_ids"] + [tokenizer.pad_token_id] * (length - len(x["input_ids"])) for x in examples]),
        "attention_mask": torch.tensor([x["attention_mask"] + [0] * (length - len(x["input_ids"])) for x in examples]),
        "labels": torch.tensor([x["labels"] + [-100] * (length - len(x["input_ids"])) for x in examples]),
    }


lengths = sorted(len(row["input_ids"]) for row in train_encoded)
stats = {
    "train_examples": len(train_encoded), "validation_examples": len(valid_encoded),
    "input_tokens": sum(lengths), "supervised_tokens": sum(sum(v != -100 for v in row["labels"]) for row in train_encoded),
    "median_length": lengths[len(lengths)//2], "max_length": max(lengths),
    "skipped_train": train_skipped, "skipped_validation": valid_skipped,
}
(SAVE_DIR / "tokenization.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
for name in ("manifest.json", "train.jsonl", "validation.jsonl", "heldout.jsonl", "tool_documentation.jsonl"):
    shutil.copy2(DATA_DIR / name, SAVE_DIR / name)
print(json.dumps(stats, indent=2), flush=True)

print("Chargement des poids Uncensored en 4 bits…", flush=True)
load_options = {
    "revision": MODEL_REVISION, "trust_remote_code": True, "token": False,
    "dtype": torch.float16, "device_map": {"": 0}, "attn_implementation": "sdpa",
}
if USE_4BIT:
    load_options["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_options)
model.config.use_cache = False
if USE_4BIT:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                          gradient_checkpointing_kwargs={"use_reentrant": False})

available_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
targets = [name for name in ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"] if name in available_names]
if len(targets) < 5:
    raise RuntimeError(f"Modules LoRA inattendus : {targets}")
model = get_peft_model(model, LoraConfig(
    r=LORA_RANK, lora_alpha=2*LORA_RANK, lora_dropout=0.05,
    target_modules=targets, task_type="CAUSAL_LM", bias="none",
))
model.print_trainable_parameters()
tokenizer.save_pretrained(SAVE_DIR / "tokenizer")

# Un vrai forward/backward, sans mise à jour des poids, avant le long traitement.
probe_record = next(row for row in train_encoded if 128 <= len(row["input_ids"]) <= 512)
probe_batch = {k: v.to("cuda") for k, v in collate([probe_record]).items()}
model.train()
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
with torch.autocast("cuda", dtype=torch.float16):
    probe_loss = model(**probe_batch).loss
if not torch.isfinite(probe_loss):
    raise RuntimeError("Perte non finie avant entraînement.")
probe_loss.backward()
gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
    raise RuntimeError("Gradients non finis avant entraînement.")
if not any(torch.count_nonzero(g).item() for g in gradients):
    raise RuntimeError("Aucun gradient LoRA non nul.")
torch.cuda.synchronize()
probe = {"loss": float(probe_loss.detach()), "seconds": round(time.perf_counter()-started, 3),
         "peak_gpu_gib": round(torch.cuda.max_memory_reserved()/2**30, 3),
         "tokens": len(probe_record["input_ids"]), "nonzero_gradients_verified": True}
(SAVE_DIR / "gpu_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
print("FORWARD/BACKWARD VALIDÉ :", json.dumps(probe), flush=True)
model.zero_grad(set_to_none=True)
del gradients, probe_batch, probe_loss
torch.cuda.empty_cache()


class Progress(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Premier point de reprise très tôt, puis sauvegardes périodiques.
        if state.global_step == 5:
            control.should_save = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        record = dict(logs or {})
        record.update(step=state.global_step, total_steps=state.max_steps,
                      gpu_gib=round(torch.cuda.memory_reserved()/2**30, 2))
        (SAVE_DIR / "progress.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print("PROGRESSION", json.dumps(record), flush=True)
        for key in ("loss", "grad_norm"):
            if key in record and not math.isfinite(float(record[key])):
                raise RuntimeError(f"Arrêt : {key} non fini")


arguments = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR), num_train_epochs=EPOCHS,
    per_device_train_batch_size=1, gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    per_device_eval_batch_size=1, learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine", warmup_steps=20,
    fp16=True, bf16=False, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch", max_grad_norm=1.0,
    logging_steps=5, logging_first_step=True, logging_nan_inf_filter=False,
    save_strategy="steps", save_steps=50, save_total_limit=3,
    eval_strategy="steps", eval_steps=100,
    report_to="none", dataloader_num_workers=0, seed=42, data_seed=42,
    remove_unused_columns=False,
)
trainer = Trainer(
    model=model, args=arguments,
    train_dataset=EncodedDataset(train_encoded), eval_dataset=EncodedDataset(valid_encoded[:64]),
    data_collator=collate, processing_class=tokenizer, callbacks=[Progress()],
)
if trainer.accelerator.scaler is not None:
    # Le T4 calcule en FP16. L'échelle par défaut 65536 déborde au premier pas
    # sur cette architecture ; le forward/backward non amplifié est déjà validé.
    trainer.accelerator.scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    print("AMP loss scale initial :", trainer.accelerator.scaler.get_scale(), flush=True)
checkpoint = get_last_checkpoint(str(CHECKPOINT_DIR))
print("Reprise :", checkpoint or "nouvel entraînement", flush=True)
print("ENTRAÎNEMENT DÉMARRÉ —", len(train_encoded), "exemples ; sauvegarde à l'étape 5 puis toutes les 50 étapes.", flush=True)
train_result = trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_model(str(SAVE_DIR / "adapter"))
tokenizer.save_pretrained(SAVE_DIR / "adapter")
trainer.save_state()
evaluation = trainer.evaluate()
results = {"training": train_result.metrics, "evaluation": evaluation,
           "peak_gpu_gib": torch.cuda.max_memory_reserved()/2**30,
           "base_model": MODEL_ID, "base_revision": MODEL_REVISION,
           "adapter_path": str(SAVE_DIR / "adapter"), "tokenization": stats}
(SAVE_DIR / "result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("ENTRAÎNEMENT TERMINÉ :", json.dumps(results, indent=2), flush=True)
