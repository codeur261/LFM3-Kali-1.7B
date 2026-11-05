# Publier ta version sur Hugging Face

L'adaptateur est le résultat de l'entraînement. Publier ce dossier suffit
pour que d'autres chargent la base et appliquent tes poids appris, sans
réentraînement. Il faut conserver la référence au modèle de base et sa licence.

## Depuis le Colab où les poids sont déjà présents

1. Crée un jeton Hugging Face avec droit d'écriture depuis
   https://huggingface.co/settings/tokens ; garde sa valeur hors du notebook.
2. Dans les secrets Colab, utilise `HF_TOKEN` et autorise ce notebook à le lire.
3. Exécute la cellule ci-dessous. Elle publie un dépôt **public** sous le pseudo
   du compte associé au jeton, avec la fiche du modèle et les poids entraînés.

```python
from pathlib import Path
import subprocess, sys
from google.colab import userdata

project = Path('/content/lfm3-github-publication')
if not project.exists():
    subprocess.run(['git', 'clone', 'https://github.com/iamarketings/LFM3-Kali-1.7B.git', str(project)], check=True)
sys.path.insert(0, str(project))
from publish_hf import publish
publish(
    '/content/drive/MyDrive/LFM3_Kali_Finetune/run_20261105_v1/adapter',
    token=userdata.get('HF_TOKEN'),
)
```

Les fichiers sont envoyés **de Colab vers Hugging Face**. Rien n'est téléchargé
sur ton ordinateur. Ce script n'a pas été exécuté pour ton compte : la publication
Hugging Face reste une action à lancer volontairement.

## Depuis ton PC

Extrais l'archive de l'adaptateur, clone ce dépôt puis exécute :

```bash
python -m pip install huggingface-hub
python publish_hf.py --adapter /chemin/vers/adapter
```

Le jeton est demandé en saisie masquée. Pour choisir un autre nom de modèle,
ajoute `--repo-id TON_PSEUDO/NOM_DU_MODELE`.

## Modèle complet ou GGUF

Cette publication distribue les adaptateurs, pas un modèle fusionné autonome.
Pour une version complète : recharger la même révision de base en FP16/BF16,
appliquer l'adaptateur, appeler `merge_and_unload()`, puis `save_pretrained()`.
Éviter de fusionner directement la base NF4 d'entraînement si l'objectif est
un export fidèle en précision complète. Une conversion GGUF compatible LFM3
constitue ensuite une étape distincte à vérifier par génération.

Références officielles :
- https://huggingface.co/docs/hub/peft
- https://huggingface.co/docs/huggingface_hub/guides/upload
- https://huggingface.co/docs/hub/model-cards
