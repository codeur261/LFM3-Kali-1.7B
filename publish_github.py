"""Regroupe le modèle et toute la formation, puis pousse directement depuis Colab."""
import getpass
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = "codeur261/LFM3-Kali-1.7B"
BASE = "OpenIntelligenceNet/LFM3-1.7B-Uncensored"
REVISION = "8e7cf5cccd95afa9d34a3a66ebfb8fef1b55c204"
RUN = Path("/content/drive/MyDrive/LFM3_Kali_Finetune/run_20261105_v1")
EXPORT = Path("/content/LFM3-Kali-1.7B-publication")


def command(*args, **kwargs):
    return subprocess.run(list(args), check=True, **kwargs)


def prepare():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not (RUN / "adapter" / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("L'adaptateur entraîné manque ; aucune relance d'entraînement automatique.")
    if not EXPORT.exists():
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
        command("git", "clone", f"https://github.com/{REPO}.git", str(EXPORT), env=env)
    command("git", "remote", "set-url", "origin", f"https://github.com/{REPO}.git", cwd=EXPORT)
    shutil.copytree(RUN, EXPORT / "training", dirs_exist_ok=True)
    shutil.copytree(RUN / "adapter", EXPORT / "adapter", dirs_exist_ok=True)
    source = Path("/content/lfm3_cyber_ft")
    if source.exists():
        for folder in ("raw", "data"):
            if (source / folder).is_dir():
                shutil.copytree(source / folder, EXPORT / "training" / folder, dirs_exist_ok=True)
    for path in (EXPORT / "adapter", EXPORT / "training" / "adapter"):
        config = json.loads((path / "adapter_config.json").read_text())
        config["revision"] = REVISION
        (path / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")

    for folder in (EXPORT / "adapter", EXPORT / "training" / "adapter"):
        shutil.copy2(EXPORT / "MODEL_CARD.md", folder / "README.md")
        shutil.copy2(EXPORT / "LICENSE-LFM", folder / "LICENSE")

    merged_dir = EXPORT / "model"
    if not (merged_dir / "EXPORT_COMPLETE.json").is_file():
        print("Fusion des poids appris dans la base originale FP16, sans réentraînement…", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE, revision=REVISION, trust_remote_code=True, token=False,
            dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa",
        )
        trained = PeftModel.from_pretrained(base, str(RUN / "adapter")).eval()
        merged = trained.merge_and_unload(safe_merge=True)
        merged.config.use_cache = True
        merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="1GB")
        tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION, trust_remote_code=True, token=False)
        tokenizer.save_pretrained(merged_dir)
        # La fiche du modèle fusionné est maintenue séparément de la fiche PEFT.
        if not (merged_dir / "README.md").is_file():
            raise FileNotFoundError("Fiche model/README.md absente de la source de publication.")
        shutil.copy2(EXPORT / "LICENSE-LFM", merged_dir / "LICENSE")
        (merged_dir / "EXPORT_COMPLETE.json").write_text(json.dumps({
            "base_model": BASE, "base_revision": REVISION,
            "source_adapter_sha256": hashlib.sha256((RUN / "adapter/adapter_model.safetensors").read_bytes()).hexdigest(),
            "format": "merged-fp16-safetensors", "max_shard_size": "1GB",
        }, indent=2) + "\n")
        del merged, trained, base
        torch.cuda.empty_cache()

    from refresh_manifest import refresh
    files = refresh(EXPORT)
    print(f"Prêt : {len(files)} fichiers, {sum(x['bytes'] for x in files)/1e9:.2f} Go", flush=True)
    for folder in ("model", "adapter", "training"):
        print(folder, sum(x["bytes"] for x in files if x["path"].startswith(folder + "/")), "octets", flush=True)
    return EXPORT


def push(token=None):
    command("git", "lfs", "version")
    if not (EXPORT / "PUBLICATION_MANIFEST.json").exists():
        raise RuntimeError("Exécuter prepare() avant push().")
    token = token or os.environ.get("GITHUB_TOKEN") or getpass.getpass(
        "Jeton GitHub pour codeur261/LFM3-Kali-1.7B (Contents: Read and write) : ")
    if not token:
        raise ValueError("Aucun jeton saisi. Aucun push effectué.")
    # Jeton éphémère en mémoire ; aucune écriture dans l'URL distante ni dans Git.
    with tempfile.TemporaryDirectory(prefix="lfm3-git-auth-") as tmp:
        askpass = Path(tmp) / "askpass.py"
        askpass.write_text("#!/usr/bin/env python3\nimport os,sys\nprint('x-access-token' if 'username' in sys.argv[1].lower() else os.environ['LFM3_GIT_AUTH'])\n")
        askpass.chmod(0o700)
        env = dict(os.environ, GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT="0", LFM3_GIT_AUTH=token)
        command("git", "lfs", "install", "--local", cwd=EXPORT)
        # Les checkpoints et shards sont des fichiers réels dans ce même dépôt ;
        # Git LFS transporte les gros binaires sans limite Git de 100 Mo.
        command("git", "lfs", "track", "*.safetensors", "*.bin", "*.pt", "*.pth", cwd=EXPORT)
        command("git", "config", "user.name", "codeur261", cwd=EXPORT)
        command("git", "config", "user.email", "186750009+codeur261@users.noreply.github.com", cwd=EXPORT)
        command("git", "add", "-f", "model", "adapter", "training", "PUBLICATION_MANIFEST.json", ".gitattributes", cwd=EXPORT)
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=EXPORT).returncode
        if staged:
            command("git", "commit", "-m", "Add trained model, adapter and complete training files", cwd=EXPORT)
        command("git", "-c", "credential.helper=", "push", "origin", "HEAD:main", cwd=EXPORT, env=env)
        command("git", "lfs", "fsck", cwd=EXPORT)
        remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"], cwd=EXPORT, env=env, text=True).split()[0]
        local = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=EXPORT, text=True).strip()
        if remote != local:
            raise RuntimeError("Le commit distant ne correspond pas au commit local.")
        print(f"PUBLICATION VÉRIFIÉE : https://github.com/{REPO}/commit/{local}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "push"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else push()
