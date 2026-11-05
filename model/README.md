---
language:
- en
- fr
license: lfm1.0
library_name: transformers
base_model: OpenIntelligenceNet/LFM3-1.7B-Uncensored
base_model_relation: finetune
pipeline_tag: text-generation
tags:
- lfm3
- lora
- qlora
- cybersecurity
- kali
---

# LFM3-Kali-1.7B — v0.1.0

Adaptateur QLoRA entraîné par Aheshman Itibar sur 5 948 exemples de commandes,
d'appels d'outils et d'explications de cybersécurité. Ce dépôt distribue les
poids de l'adaptateur réellement entraîné, pas le modèle de base complet.

Base : `OpenIntelligenceNet/LFM3-1.7B-Uncensored`, révision immuable
`8e7cf5cccd95afa9d34a3a66ebfb8fef1b55c204`, dérivée de `LiquidAI/LFM3-1.7B`.
Les modifications apprises sont distribuées dans `adapter_model.safetensors`.
La licence du modèle d'origine, LFM Open License v1.0, reste applicable.

## Données et paramètres

| Source | Entraînement |
| --- | ---: |
| KaliBench | 3 329 |
| Cyber-SFT-Agent, premier appel uniquement | 1 583 |
| HackMentor, explications | 932 |
| Gabarits français | 104 |

313 exemples de validation, 5 294 exemples séparés réservés à l'évaluation.
QLoRA NF4, rang 16, alpha 32, dropout 0,05, une époque, accumulation 8,
taux initial 1e-4, FP16, 16 105 472 paramètres entraînables. 744 étapes en
3 553,3157 s sur Tesla T4 ; 1 365 622 tokens d'entrée, 283 503 tokens supervisés.

## Évaluation et limites observées

La perte finale 0,008758 est calculée sur les **64 premiers exemples** de
validation, un sous-ensemble non représentatif. Elle ne prouve pas la qualité
conversationnelle. Aucun score d'exécution KaliBench global n'est annoncé.

La première comparaison améliore le scan Nmap TCP connect et le filtre DNS
tshark. D'autres tests produisent `nmap -sV` pour afficher la version du binaire,
des définitions incorrectes des états de ports et des appels d'outils à la place
d'explications. La version v0.1.0 reste à évaluer et améliorer.

MSF est peu représenté : pas de parcours pratique de msfconsole/msfvenom.
Le corpus n'enseigne pas de boucle complète d'exécution et de retour d'outil.
Les commandes générées pendant les tests n'ont pas été exécutées.
Aucun corpus de refus n'a été ajouté ; aucun comportement sans refus n'est garanti.

## Recharger sans entraîner

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "OpenIntelligenceNet/LFM3-1.7B-Uncensored"
REV = "8e7cf5cccd95afa9d34a3a66ebfb8fef1b55c204"
ADAPTER = "./adapter"  # ou TON_PSEUDO/LFM3-Kali-1.7B sur Hugging Face
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForCausalLM.from_pretrained(
    BASE, revision=REV, trust_remote_code=True,
    dtype=torch.float16, device_map={"": 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    ),
)
model = PeftModel.from_pretrained(base, ADAPTER).eval()
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Avec tshark, lis capture.pcap et affiche les paquets DNS."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    result = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id, use_cache=True)
print(tokenizer.decode(result[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

Environnement utilisé : torch 2.11.0, transformers 5.20.2, peft 0.21.2,
accelerate 1.15.0, bitsandbytes 0.50.2. Dépôt du code et des sources :
https://github.com/iamarketings/LFM3-Kali-1.7B

Sources : RISys-Lab/KaliBench (CC BY 4.0), hotdogs/cyber-sft-agent-qwen38
(Apache 2.0), tmylla/HackMentor (dépôt Apache 2.0). Les révisions, empreintes
et filtres de sélection sont publiés avec le code et le manifeste.


Ce sous-dossier contient le modèle complet fusionné en FP16. Les adaptateurs d'origine restent disponibles dans ../adapter.
