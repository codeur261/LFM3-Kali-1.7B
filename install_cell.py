import platform
import subprocess
import sys

print("Python :", platform.python_version())
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], check=True)

import torch

print("PyTorch :", torch.__version__, "CUDA :", torch.version.cuda)
if not torch.cuda.is_available():
    raise RuntimeError("Sélectionner GPU T4 dans Exécution > Modifier le type d'exécution.")
print("GPU détecté :", torch.cuda.get_device_name(0))
print("Installation des bibliothèques de fine-tuning…", flush=True)

# Dans le notebook, cette ligne est convertie en %pip.
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0",
    "bitsandbytes==0.50.2", "huggingface-hub==2.3.0",
    "tokenizers==0.23.3", "safetensors==0.8.0", "datasets==5.1.0",
], check=True)
print("Installation terminée. Passer à la préparation des données.")
