# Publication sur Hugging Face

Le script publie les poids de l'adaptateur entraîné avec sa configuration, son tokenizer, sa fiche et la licence du modèle. Le compte de destination est obtenu avec HfApi.whoami() à partir du jeton fourni : le nom du compte GitHub n'est pas supposé être celui du compte Hugging Face.

## Préparation

Créer un jeton d'écriture depuis https://huggingface.co/settings/tokens et le conserver dans le secret Colab HF_TOKEN ou dans une saisie masquée. La valeur ne doit pas être intégrée au notebook ou à un commit.

Le script crée les **nouveaux dépôts en privé par défaut**. Ce paramètre ne modifie pas la visibilité d'un dépôt existant. Après vérification du chargement de l'adaptateur et d'une génération, la visibilité peut être changée dans les réglages Hugging Face. Une création directement publique reste possible avec private=False ou l'option CLI --public.

## Depuis Colab

Le code de publication provient du dépôt **codeur261/LFM3-Kali-1.7B**. Vérifier la provenance et la révision du code avant de lui donner accès à un jeton. Un ancien clone doit être contrôlé ou remplacé avant exécution.

~~~python
from pathlib import Path
import subprocess, sys
from google.colab import userdata

project = Path("/content/lfm3-hf-publication")
expected = "https://github.com/codeur261/LFM3-Kali-1.7B.git"
if not project.exists():
    subprocess.run(["git", "clone", expected, str(project)], check=True)
origin = subprocess.check_output(
    ["git", "-C", str(project), "remote", "get-url", "origin"], text=True,
).strip()
if origin != expected:
    raise RuntimeError("Origine du code inattendue : vérifier le clone avant publication.")
sys.path.insert(0, str(project))
from publish_hf import publish
publish(
    "/content/drive/MyDrive/LFM3_Kali_Finetune/run_20261105_v1/adapter",
    token=userdata.get("HF_TOKEN"),
    private=True,
)
~~~

L'envoi s'effectue directement de Colab vers Hugging Face. Le nom du dépôt est construit avec le compte HF authentifié, sans identifiant utilisateur fictif.

## Depuis un PC

~~~bash
python -m pip install huggingface-hub
python publish_hf.py --adapter ./adapter
~~~

Le jeton est demandé en saisie masquée. L'option --repo-id permet de préciser un dépôt appartenant au compte authentifié ; --public permet une création publique explicite.

## Vérifier avant ouverture au public

Contrôler la présence des poids, de la configuration, du tokenizer, du README et de la licence. Charger la base à la révision indiquée dans MODEL_CARD.md, puis l'adaptateur avec PeftModel.from_pretrained(), et effectuer une génération. Le contrôle doit porter sur les fichiers effectivement publiés et leur révision.

## Modèle complet

Le dossier model/ du dépôt GitHub contient déjà le modèle fusionné FP16. Il peut être publié dans un dépôt HF distinct avec HfApi.upload_folder(), accompagné de son README, de sa licence, de sa configuration, de son code et de ses shards. Le script publish_hf.py présenté ici publie uniquement l'adaptateur.

Le modèle complet et l'adaptateur sont deux formats de distribution du même entraînement. La conversion GGUF est une étape distincte, à vérifier avec un backend compatible LFM3.

Références :
- https://huggingface.co/docs/hub/peft
- https://huggingface.co/docs/huggingface_hub/guides/upload
- https://huggingface.co/docs/hub/model-cards
