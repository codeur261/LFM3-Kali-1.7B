# Paramètres de la première version. Le contexte d'entraînement est distinct du contexte d'inférence.
import os
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive", force_remount=False)

WORK_DIR = Path("/content/lfm3_cyber_ft")
SAVE_DIR = Path("/content/drive/MyDrive/LFM3_Kali_Finetune/run_20261105_v1")
WORK_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["LFM3_WORK_DIR"] = str(WORK_DIR)
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

MODEL_ID = "OpenIntelligenceNet/LFM3-1.7B-Uncensored"
MODEL_REVISION = "8e7cf5cccd95afa9d34a3a66ebfb8fef1b55c204"
MAX_LENGTH = 1536
LORA_RANK = 16
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION = 8
EPOCHS = 1
USE_4BIT = True
print("Dossier des sauvegardes :", SAVE_DIR)
print("Modèle :", MODEL_ID)
