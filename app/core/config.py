from pathlib import Path

# Root of the project (2 levels up from app/core/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODELS_ONNX_DIR = PROJECT_ROOT / "models_onnx"
SAM2_CONFIGS_DIR = PROJECT_ROOT / "sam2" / "configs"
