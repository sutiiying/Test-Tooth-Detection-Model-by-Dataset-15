from pathlib import Path
from huggingface_hub import hf_hub_download

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

path = hf_hub_download(
    repo_id="nsitnov/8024-yolov8-model",
    filename="8024.pt",
    local_dir=str(MODEL_DIR),
)
print(f"Model downloaded to: {path}")
