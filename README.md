# LFM3-Kali-1.7B

### Un modèle compact spécialisé dans les outils de cybersécurité

LFM3-Kali-1.7B est un projet d'adaptation d'un modèle de langage de 1,7 milliard de paramètres à la génération de commandes Kali Linux et aux appels d'outils structurés. Le projet réunit les poids entraînés, un corpus traçable et un pipeline reproductible de fine-tuning QLoRA.

La version **v0.1.0** est une première version expérimentale ouverte aux évaluations et aux contributions.

[Fiche du modèle](MODEL_CARD.md) · [Contribuer](CONTRIBUTING.md) · [Publication Hugging Face](PUBLISH_HUGGINGFACE.md)

## Présentation

| Élément | Description |
| --- | --- |
| Modèle de départ | [OpenIntelligenceNet/LFM3-1.7B-Uncensored](https://huggingface.co/OpenIntelligenceNet/LFM3-1.7B-Uncensored), dérivé de LiquidAI/LFM3-1.7B |
| Méthode | QLoRA NF4, adaptateurs de rang 16 |
| Données d'entraînement | 5 948 exemples sélectionnés et dédupliqués |
| Formats | Modèle complet fusionné FP16 et adaptateur PEFT |
| Langues des requêtes | Anglais et français |
| Environnement de référence | Google Colab, NVIDIA Tesla T4 |

La révision de base est fixée à **8e7cf5cccd95afa9d34a3a66ebfb8fef1b55c204** pour assurer la reproductibilité.

## Organisation du dépôt

Le modèle et les fichiers de formation sont regroupés dans ce dépôt. Les binaires volumineux sont gérés avec **Git LFS**.

| Emplacement | Contenu |
| --- | --- |
| model/ | Modèle complet fusionné, poids Safetensors, configuration et tokenizer |
| adapter/ | Poids LoRA entraînés et configuration PEFT |
| training/ | Checkpoints, état de l'optimiseur, données préparées, sources brutes et résultats |
| data/manifest.json | Origine, révisions et empreintes des données |
| LFM3_Kali_QLoRA_Training.ipynb | Notebook d'entraînement Google Colab |
| prepare_data.py | Préparation et filtrage du corpus |
| train_model.py | Entraînement QLoRA et reprise des checkpoints |
| test_after_training.py | Comparaison qualitative avec le modèle de base |
| publish_github.py | Export et publication depuis Colab |
| publish_hf.py | Publication de l'adaptateur sur Hugging Face |
| PUBLICATION_MANIFEST.json | Inventaire et empreintes SHA-256 des artefacts |

## Démarrage rapide

### Récupérer les poids

Installer [Git LFS](https://git-lfs.com/), puis cloner le dépôt. Le téléchargement sélectif récupère uniquement le modèle complet ; les fichiers de formation restent disponibles séparément.

~~~bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/iamarketings/LFM3-Kali-1.7B.git
cd LFM3-Kali-1.7B
git lfs install --local
git lfs pull --include="model/**"
~~~

Le modèle complet occupe environ **3,48 Go sur disque**. La mémoire d'inférence dépend de la précision, du contexte et du backend.

### Générer une réponse

L'exemple suivant utilise un GPU NVIDIA et un chargement en 4 bits. PyTorch doit être installé avec une version CUDA compatible avec le matériel utilisé.

~~~bash
python -m pip install "transformers==5.20.2" "accelerate==1.15.0" "bitsandbytes==0.50.2"
~~~

~~~python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map={"": 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    ),
).eval()

messages = [{
    "role": "user",
    "content": "Avec tshark, lis capture.pcap et affiche les paquets DNS.",
}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
with torch.inference_mode():
    generated = model.generate(
        **inputs, max_new_tokens=160, do_sample=False, use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
~~~

Ce programme génère une réponse textuelle ; il n'exécute pas les commandes proposées. Le chargement de l'adaptateur sur la base d'origine est décrit dans la [fiche du modèle](MODEL_CARD.md).

## Données

| Source | Exemples | Utilisation |
| --- | ---: | --- |
| [KaliBench](https://github.com/RISys-Lab/KaliBench) | 3 329 | Requêtes et commandes Kali/Linux |
| [Cyber-SFT-Agent](https://huggingface.co/datasets/hotdogs/cyber-sft-agent-qwen38) | 1 583 | Premier appel d'outil et arguments structurés |
| [HackMentor](https://github.com/tmylla/HackMentor) | 932 | Explications de cybersécurité |
| Gabarits français du projet | 104 | Commandes et requêtes de base |
| **Total** | **5 948** | |

Le corpus comprend également **313 exemples de validation** et **5 294 exemples réservés à l'évaluation**. Les requêtes identiques sont exclues entre les ensembles. Pour Cyber-SFT-Agent, seul le premier appel d'outil est retenu, après vérification du schéma et des arguments ; les sorties synthétiques et les raisonnements intermédiaires sont exclus.

Une collection distincte de **2 809 extraits de documentation** est fournie pour des travaux de recherche documentaire. Elle n'est pas utilisée comme un corpus de résultats d'exécution. Les sources, licences et empreintes sont consignées dans le [manifeste](data/manifest.json).

## Reproduire l'entraînement

Ouvrir [le notebook dans Google Colab](https://colab.research.google.com/github/iamarketings/LFM3-Kali-1.7B/blob/main/LFM3_Kali_QLoRA_Training.ipynb), sélectionner un GPU et exécuter les cellules dans l'ordre. Le pipeline vérifie le matériel et les gradients avant le calcul. Les checkpoints permettent une reprise après interruption.

| Paramètre | Valeur |
| --- | --- |
| Adaptateurs | LoRA, rang 16, alpha 32, dropout 0,05 |
| Paramètres entraînables | 16 105 472 |
| Quantification | NF4, double quantification |
| Calcul | FP16, gradient checkpointing |
| Batch / accumulation | 1 / 8 |
| Taux d'apprentissage initial | 1 × 10⁻⁴ |
| Durée | 1 époque, 744 étapes |
| Limite de séquence | 1 536 tokens |
| Tokens d'entrée / supervisés | 1 365 622 / 283 503 |

Environnement de référence : PyTorch 2.11.0, Transformers 5.20.2, PEFT 0.21.2, Accelerate 1.15.0 et BitsAndBytes 0.50.2.

## Évaluation

L'entraînement de référence a duré **59 min 13 s** sur Tesla T4, avec une mémoire GPU réservée maximale de **6,50 Gio**. La perte moyenne d'entraînement est de 0,542719 et la perte finale de validation de 0,008758.

Cette dernière mesure porte sur les **64 premiers exemples de validation** et ne constitue pas un score global de compétence. Les tests qualitatifs montrent des améliorations sur certaines commandes Nmap et tshark, ainsi que des erreurs persistantes d'interprétation, de syntaxe et de choix du format de réponse. Les rapports bruts sont conservés dans **training/**.

La version v0.1.0 ne dispose pas encore d'un score d'exécution global sur le jeu de test réservé. Les commandes produites lors des comparaisons qualitatives n'ont pas été exécutées.

## Périmètre et limites

- La spécialisation concerne principalement la proposition de commandes et le premier appel d'outil.
- Le corpus ne couvre pas une boucle agent complète avec exécution, retour d'outil et correction.
- Metasploit est faiblement représenté ; les parcours pratiques msfconsole et msfvenom restent à développer.
- Le corpus français est limité et ne constitue pas une évaluation linguistique complète.
- Aucun corpus de refus supplémentaire n'a été ajouté ; le comportement dépend néanmoins de la base et du contexte.
- La conversion GGUF et les mesures sur des configurations locales constituent des travaux distincts.

## Contribuer

Les contributions sont ouvertes via les [issues](https://github.com/iamarketings/LFM3-Kali-1.7B/issues) et les [pull requests](https://github.com/iamarketings/LFM3-Kali-1.7B/pulls). Les priorités concernent l'évaluation indépendante, la qualité des données, la couverture des outils et les réponses en français.

Les critères de provenance, de licence et de validation sont décrits dans [CONTRIBUTING.md](CONTRIBUTING.md).

## Licences et crédits

- **Code original et documentation du projet** : [licence MIT](LICENSE).
- **Poids du modèle et dérivés** : [LFM Open License v1.0](LICENSE-LFM).
- **Données et documentation tierces** : licences propres aux sources, détaillées dans le manifeste.

Le projet s'appuie sur les travaux de [Liquid AI](https://huggingface.co/LiquidAI), [OpenIntelligenceNet](https://huggingface.co/OpenIntelligenceNet), [KaliBench](https://github.com/RISys-Lab/KaliBench), [HackMentor](https://github.com/tmylla/HackMentor) et [Cyber-SFT-Agent](https://huggingface.co/datasets/hotdogs/cyber-sft-agent-qwen38). La recette d'entraînement s'appuie également sur le [cookbook Liquid4All](https://github.com/Liquid4All/cookbook).

**Maintenance : [Aheshman Itibar — iamarketings](https://github.com/iamarketings).**
