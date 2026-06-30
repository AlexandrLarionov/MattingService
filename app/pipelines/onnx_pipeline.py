import numpy as np
import onnxruntime as ort
from PIL import Image, ExifTags
from pathlib import Path
from loguru import logger

BIREFNET_SIZE = (1024, 1024)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _fix_exif(img: Image.Image) -> Image.Image:
    try:
        exif = img.getexif()
        if not exif:
            return img
        for tag, name in ExifTags.TAGS.items():
            if name == "Orientation":
                orientation = exif.get(tag)
                break
        else:
            return img
        ops = {
            2: Image.Transpose.FLIP_LEFT_RIGHT,
            3: Image.Transpose.ROTATE_180,
            4: Image.Transpose.FLIP_TOP_BOTTOM,
            5: Image.Transpose.TRANSPOSE,
            6: Image.Transpose.ROTATE_270,
            7: Image.Transpose.TRANSVERSE,
            8: Image.Transpose.ROTATE_90,
        }
        if orientation in ops:
            img = img.transpose(ops[orientation])
    except Exception:
        pass
    return img

def load_image(source) -> Image.Image:
    if isinstance(source, Image.Image):
        img = source.copy()
    else:
        img = Image.open(source)
    img = _fix_exif(img)
    return img.convert("RGB")

def preprocess_rgb(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    resized = image.resize(size, Image.Resampling.LANCZOS)
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0).astype(np.float32)

def postprocess_mask(raw: np.ndarray) -> np.ndarray:
    return np.clip(raw.squeeze(), 0.0, 1.0).astype(np.float32)

def _to_pil_mask(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8), mode="L")

class OnnxModel:
    def __init__(self, model_path, providers=None):
        self.path = Path(model_path)
        if not self.path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.path}")
        self.session = ort.InferenceSession(
            str(self.path), providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"Loaded ONNX model: {self.path.name}")

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: tensor})[0]

class ONNXPipeline:
    def __init__(self, model_path, providers=None):
        logger.info(f"Initializing BiRefNet from: {model_path}")
        self._model = OnnxModel(model_path, providers)

    def process_image(self, image_source) -> Image.Image:
        rgb = load_image(image_source)
        
        # Step 1: Preprocess
        tensor = preprocess_rgb(rgb, BIREFNET_SIZE)
        
        # Step 2: Inference
        alpha_raw = postprocess_mask(self._model(tensor))
        
        # Step 3: Postprocess
        mask = _to_pil_mask(alpha_raw)
        
        # Combine
        rgba = rgb.convert("RGBA")
        rgba.putalpha(mask.resize(rgb.size, Image.Resampling.LANCZOS))
        return rgba
