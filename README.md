# LFM3 — spécialisation aux outils Kali

**Version v0.1.0 : entraînement terminé, résultats qualitatifs encore imparfaits.**
5 948 exemples, 744 étapes sur Tesla T4, 59 min 13 s. Les commandes Nmap TCP
connect et le filtre DNS tshark s'améliorent dans les premiers tests. Des
explications restent fausses et la version de Nmap est parfois confondue avec
la détection des versions de services. La faible perte de validation porte sur
les 64 premiers exemples de validation : elle ne mesure pas une compétence
générale. Cette version est publiée pour permettre les évaluations et contributions.

- [Guide de publication Hugging Face](PUBLISH_HUGGINGFACE.md)
- [Fiche du modèle](MODEL_CARD.md)
- [Contribuer](CONTRIBUTING.md)
- [Versions et poids entraînés](https://github.com/iamarketings/LFM3-Kali-1.7B/releases)

La publication complète est regroupée dans **ce même dépôt** : `model/` pour
les poids complets fusionnés, `adapter/` pour l'adaptateur appris et `training/`
pour les données, fichiers bruts, checkpoints, optimiseur, paramètres et résultats.
Les gros fichiers sont transportés par Git LFS. Le script `publish_github.py`
prépare ce dossier puis le pousse directement depuis Colab. Un dossier n'est
disponible à distance qu'une fois le push vérifié, pas dès sa préparation.

Le modèle entraîné est un adaptateur PEFT : `adapter_model.safetensors`
contient réellement les poids appris, à charger avec la révision de base ci-dessous.
Il ne faut pas relancer l'entraînement pour l'utiliser. La fusion en un modèle
autonome et la conversion GGUF sont des étapes distinctes.

Modèle de départ : `OpenIntelligenceNet/LFM3-1.7B-Uncensored`, révision
`8e7cf5cccd95afa9d34a3a66ebfb8fef1b55c204`.

Exécuter les cellules dans l'ordre avec un GPU T4. Google Drive conserve les
données préparées, les adaptateurs, l'état de l'optimiseur et les résultats.
La cellule d'entraînement détecte automatiquement un checkpoint pour reprendre.
Le téléchargement initial des poids est d'environ 3,46 Go.

## Corpus préparé

| Source | Exemples d'entraînement | Usage |
| --- | ---: | --- |
| KaliBench, GitHub RISys-Lab | 3 329 | Demande vers commande Kali/Linux |
| Cyber-SFT-Agent | 1 583 | Premier appel d'outil et arguments validés |
| HackMentor, GitHub tmylla | 932 | Explications de cybersécurité |
| Gabarits français documentés | 104 | Demandes françaises et commandes de base |
| **Total** | **5 948** | |

313 exemples de validation et 5 294 exemples réservés à l'évaluation sont
séparés. Aucun exemple du test original KaliBench n'entre dans l'entraînement.
Les requêtes identiques sont dédupliquées entre les ensembles. Les sorties
synthétiques et raisonnements de Cyber-SFT-Agent sont exclus ; le premier
appel d'outil est conservé seulement si son schéma et ses arguments concordent.
La validation syntaxique n'est pas une preuve que toute commande fonctionnera
dans une installation Kali donnée.

Les 2 809 extraits de documentation d'outils sont livrés séparément pour un RAG
ultérieur. Ils ne sont pas présentés comme des résultats d'exécution.

## Entraînement

- QLoRA NF4, rang 16, alpha 32, dropout 0,05.
- 16 105 472 paramètres entraînables, soit 0,923 % du modèle chargé.
- Batch 1, accumulation 8, une époque, taux d'apprentissage 0,0001.
- Limite 1 536 tokens. Le plus long exemple préparé mesure 549 tokens avec le
  tokenizer utilisé ; aucun exemple n'a dû être tronqué ou écarté pour sa taille.
- 1 365 622 tokens d'entrée et 283 503 tokens assistant supervisés.
- Format natif LFM3, appels d'outils structurés et mode thinking désactivé.
- T4 : calcul FP16, adaptateurs FP32, gradient checkpointing ; l'échelle AMP
  initiale est réduite à 128 pour éviter le débordement observé à 65 536.
- Contrôle forward/backward réel avant le long calcul ; arrêt sur perte ou
  gradients non finis.
- Sauvegarde à l'étape 5, puis toutes les 50 étapes ; trois points conservés.

L'entraînement complet a été exécuté sur Tesla T4 : 744 étapes, 3 553,3157 s,
perte moyenne 0,542719 et mémoire réservée maximale 6,498 Gio. La perte finale
sur le sous-ensemble de validation est 0,008758. Les contrôles qualitatifs
révèlent des erreurs malgré cette perte faible ; aucun score global de maîtrise
des outils n'est revendiqué.

Aucun corpus de refus ou d'alignement supplémentaire n'est ajouté. Cela ne
garantit pas l'absence totale de refus, d'erreurs ou de régressions. Le contrôle
qualitatif final compare le modèle de départ et l'adaptateur sur les mêmes
questions, sans exécuter les commandes proposées.

## Résultats

`Google Drive/MyDrive/LFM3_Kali_Finetune/run_20261105_v1/` contient :

- `manifest.json` et les données préparées ;
- `gpu_probe.json`, `tokenization.json`, `progress.json` ;
- `checkpoints/checkpoint-*/` pour la reprise ;
- `adapter/` après entraînement, `result.json` et `qualitative_tests.json`.

Le dossier `adapter/` doit être utilisé avec la même révision du modèle de base.
La fusion puis la conversion GGUF sont des opérations distinctes pour le
déploiement local ; ce dossier contient les adaptateurs d'entraînement.

## Sources et attribution

- Pengfei Li, Naufal Suryanto, Sicheng Zhang, Muzammal Naseer, **KaliBench**,
  https://github.com/RISys-Lab/KaliBench — données vérifiées CC BY 4.0.
- Jie Zhang, Hui Wen, Liting Deng, Mingfeng Xin, Zhi Li, Lun Li, Hongsong Zhu,
  Limin Sun, **HackMentor: Fine-tuning Large Language Models for Cybersecurity**,
  IEEE TrustCom 2023 — https://github.com/tmylla/HackMentor, dépôt Apache 2.0.
- **Cyber-SFT-Agent**, hotdogs, v3.1,
  https://huggingface.co/datasets/hotdogs/cyber-sft-agent-qwen38 — Apache 2.0.
- Recette de référence LFM3 SFT : https://github.com/Liquid4All/cookbook.

Les URL exactes et SHA-256 de chaque fichier sont inscrits dans le manifeste.
Le corpus restreint CyberAgentTrace n'est pas utilisé.

MSF est peu couvert : 21 exemples de commandes liés à MSF/Armitage et quatre
mentions théoriques, sans exemples pratiques assistant `msfconsole` ou
`msfvenom`. Les définitions d'outils présentes dans une palette ne sont pas
comptées comme exemples d'utilisation.

Le code original du projet est sous licence MIT (`LICENSE`). Les poids
dérivent de LiquidAI et restent soumis à la LFM Open License v1.0
(`LICENSE-LFM`). Les jeux de données conservent leurs licences et attributions
propres, détaillées dans `data/manifest.json`.

## Sauvegarde GitHub

Le dépôt privé peut conserver ce code, le notebook et les fichiers JSONL du
dossier `data/`, avec leurs sources et empreintes. Les fichiers d'entraînement
bruts sont récupérables depuis le manifeste et ne sont pas dupliqués dans Git.

Les adaptateurs LoRA et les archives de checkpoints doivent être attachés aux
GitHub Releases du projet, plutôt qu'ajoutés à chaque commit. Les poids du
modèle de départ sont téléchargés depuis leur dépôt public à la réutilisation.
L'entraînement actuellement lancé conserve ses checkpoints sur Drive ; aucun
de ces fichiers ne doit être effacé avant la vérification de sa copie distante.
