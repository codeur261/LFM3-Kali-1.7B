# LFM3-Kali — Ollama / GGUF Q4_K_M

Modèle complet fine-tuné, fusionné puis quantifié en Q4_K_M. Aucun adaptateur externe à charger.

## Installation locale

Installer ou mettre à jour Ollama depuis https://ollama.com/download.
Télécharger dans le même dossier :
- [LFM3-Kali-1.7B-Q4_K_M.gguf](https://huggingface.co/aheshman/LFM3-Kali-1.7B/resolve/main/gguf/LFM3-Kali-1.7B-Q4_K_M.gguf?download=true)
- [Modelfile](https://huggingface.co/aheshman/LFM3-Kali-1.7B/resolve/main/gguf/Modelfile?download=true)

Ouvrir un terminal dans ce dossier :

~~~bash
ollama create lfm3-kali -f Modelfile
ollama run lfm3-kali
~~~

Pour quitter le chat : /bye. Ollama utilise par défaut le contexte du Modelfile : 4096 tokens. Le dialogue du chat utilise cette fenêtre ; les échanges les plus anciens peuvent être tronqués lorsqu’elle est remplie. Pour un historique durable, utiliser une interface qui enregistre les conversations.

Commencer à 4096 tokens sur 4 Go de VRAM. Augmenter num_ctx progressivement selon la mémoire libre ; 64K n’est pas une garantie de tenue en VRAM. Les outils Kali doivent être installés séparément. Le chat propose des commandes ; aucun outil système n’est exécuté automatiquement.

## Vérification

- Taille : 1099627040 octets.
- SHA-256 : f8c710c66c3b70f6dfa49fee8cced077c61496eca6ae28d5e8d67567c5216cde.
- Import et génération testés avec Ollama 0.40.14, CPU Colab, contexte 4096.
- Recette : modèle FP16 publié → llama.cpp F16 GGUF → Q4_K_M.
- llama.cpp : 2ada83079afc79295b3c07bae8a74457c61cba92.
- Résultats bruts : validation.json. Ces tests ne constituent pas un benchmark ni une mesure sur le PC de l’utilisateur.

La version reste expérimentale. Licence des poids : LFM Open License v1.0 ; consulter le fichier LICENSE à la racine.

## Limite observée pendant le test

Le modèle répond en français et génère des commandes, mais le test tshark a produit une syntaxe incorrecte. Le détail figure dans validation.json. La compatibilité Ollama est vérifiée ; la justesse des commandes doit être contrôlée.
