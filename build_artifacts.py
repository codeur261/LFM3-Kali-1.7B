import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def code(text, cell_id):
    return {"cell_type": "code", "id": cell_id, "execution_count": None,
            "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


installation = read("install_cell.py")
start = installation.index("subprocess.run([\n    sys.executable")
end = installation.index('print("Installation terminée.', start)
installation = installation[:start] + '%pip install -q "transformers==5.20.2" "peft==0.21.2" "accelerate==1.15.0" "bitsandbytes==0.50.2" "huggingface-hub==2.3.0" "tokenizers==0.23.3" "safetensors==0.8.0" "datasets==5.1.0"\n' + installation[end:]
intro = """# LFM3 — fine-tuning Kali sur T4

Base : **OpenIntelligenceNet/LFM3-1.7B-Uncensored**. Corpus : **5 948 exemples**,
313 exemples de validation et un ensemble de test séparé. QLoRA 4 bits, rang 16.

Exécuter les cellules dans l'ordre. Autoriser le montage de Google Drive pour
conserver les sauvegardes. La cellule d'entraînement reprend automatiquement
le dernier checkpoint de `MyDrive/LFM3_Kali_Finetune/run_20261105_v1`.

La première exécution télécharge environ 3,46 Go de poids. Le calcul est un
fine-tuning des adaptateurs ; seuls les tokens assistant sont supervisés.
Les commandes des datasets ne sont jamais exécutées. La dernière cellule
compare les réponses du modèle de base et du modèle spécialisé.

Une fermeture ou déconnexion peut interrompre la session Colab ; les checkpoints
sur Drive permettent de reprendre. Aucun résultat final n'est prérempli ici.
"""
notebook = {
    "cells": [{"cell_type": "markdown", "id": "objectif", "metadata": {}, "source": intro.splitlines(keepends=True)},
              code(installation, "installation"), code(read("config_cell.py"), "configuration-drive"),
              code(read("prepare_data.py"), "corpus"), code(read("train_model.py"), "entrainement"),
              code(read("test_after_training.py"), "comparaison")],
    "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
path = ROOT / "LFM3_Kali_QLoRA_Training.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        ordinary = "\n".join(line for line in "".join(cell["source"]).splitlines() if not line.startswith("%"))
        compile(ordinary, cell["id"], "exec")
bundle = ROOT / "LFM3_Kali_Corpus_et_Code.zip"
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for name in ["README.md", "prepare_data.py", "train_model.py", "config_cell.py", "install_cell.py", "test_after_training.py", path.name]:
        archive.write(ROOT / name, name)
    for item in sorted((ROOT / "data").iterdir()):
        if item.is_file():
            archive.write(item, "data/" + item.name)
print(json.dumps({"notebook": str(path), "notebook_bytes": path.stat().st_size,
                  "bundle": str(bundle), "bundle_bytes": bundle.stat().st_size, "cells": len(notebook["cells"])}, indent=2))
