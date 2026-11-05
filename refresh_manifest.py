"""Régénère le manifeste des artefacts sans télécharger les objets LFS absents."""
import argparse
import hashlib
import json
import re
from pathlib import Path

LFS_POINTER = re.compile(
    rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?"
)

def artifact_entry(path, root):
    with path.open("rb") as stream:
        prefix = stream.read(1024)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            match = LFS_POINTER.fullmatch(prefix)
            if match is None:
                raise ValueError(f"Pointeur LFS invalide : {path}")
            size, digest = int(match[2]), match[1].decode("ascii")
        else:
            stream.seek(0)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            size = path.stat().st_size
    return {"path": path.relative_to(root).as_posix(), "bytes": size, "sha256": digest}

def build_manifest(root):
    root = Path(root).resolve()
    entries = []
    for folder in ("model", "adapter", "training"):
        directory = root / folder
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    entries.append(artifact_entry(path, root))
    if not entries:
        raise ValueError("Aucun artefact model/, adapter/ ou training/ trouvé.")
    return entries

def refresh(root, check=False):
    root = Path(root).resolve()
    entries = build_manifest(root)
    output = root / "PUBLICATION_MANIFEST.json"
    if check:
        if not output.exists() or json.loads(output.read_text(encoding="utf-8")) != entries:
            raise ValueError("Le manifeste ne correspond pas aux artefacts.")
    else:
        output.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"{len(entries)} artefacts ; {sum(x['bytes'] for x in entries)} octets")
    return entries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args()
    refresh(options.root, options.check)
