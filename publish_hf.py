"""Publie l'adaptateur déjà entraîné, sans relancer ni fusionner le modèle."""
import argparse
import getpass
import json
import os
import tempfile
import shutil
from pathlib import Path


def publish(adapter_dir, repo_id=None, token=None):
    from huggingface_hub import HfApi

    adapter_dir = Path(adapter_dir)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        if not (adapter_dir / name).is_file():
            raise FileNotFoundError(adapter_dir / name)
    token = token or os.environ.get("HF_TOKEN") or getpass.getpass("Jeton Hugging Face avec droit d'écriture : ")
    api = HfApi(token=token)
    username = api.whoami()["name"]
    repo_id = repo_id or f"{username}/LFM3-Kali-1.7B"
    card = Path(__file__).with_name("MODEL_CARD.md")
    license_path = Path(__file__).with_name("LICENSE-LFM")
    with tempfile.TemporaryDirectory(prefix="lfm3-hf-") as temporary:
        staging = Path(temporary)
        for name in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json",
                     "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
            source = adapter_dir / name
            if source.is_file():
                shutil.copy2(source, staging / name)
        config = json.loads((staging / "adapter_config.json").read_text())
        config["revision"] = "8e7cf5cccd95afa9d34a3a66ebfb8fef1b55c204"
        (staging / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
        shutil.copy2(card, staging / "README.md")
        shutil.copy2(license_path, staging / "LICENSE")
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(staging),
                                   commit_message="Publish trained LFM3 Kali QLoRA adapter v0.1.0")
    print(f"Publié : https://huggingface.co/{repo_id}")
    print(f"Commit : {commit.oid}")
    return repo_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--repo-id")
    args = parser.parse_args()
    publish(args.adapter, args.repo_id)
