import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import onnxruntime
from pathlib import Path
from loguru import logger

# Config
from app.core.config import MODELS_ONNX_DIR, SAM2_CONFIGS_DIR

# SAM2 Imports
from sam2.build_sam import build_sam2matting
from sam2.sam2matting_image_predictor import SAM2MattingImagePredictor

class MattingPipeline:
    def __init__(self, 
                 ben2_onnx_path=None, 
                 sam2_checkpoint=None, 
                 sam2_cfg="sam2matting-sam2.1base+.yaml", 
                 device=None):
        
        import hydra
        from hydra.core.global_hydra import GlobalHydra
        
        ben2_onnx_path = ben2_onnx_path or str(MODELS_ONNX_DIR / "BEN2_Base.onnx")
        sam2_checkpoint = sam2_checkpoint or str(MODELS_ONNX_DIR / "checkpoints" / "SAM2Matting-SAM2.1Base2B.pt")
        config_dir = str(SAM2_CONFIGS_DIR)
        
        GlobalHydra.instance().clear()
        hydra.initialize_config_dir(config_dir=config_dir, version_base=None)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Initializing SAM2 MattingPipeline on device: {self.device}")

        logger.info("Loading BEN2...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.ben2_session = onnxruntime.InferenceSession(ben2_onnx_path, providers=providers)
        
        logger.info("Loading SAM2Matting...")
        self.sam2_predictor = SAM2MattingImagePredictor(build_sam2matting(sam2_cfg, sam2_checkpoint, device=self.device))
        logger.info("SAM2 Pipeline initialized successfully.\n")

    def _preprocess_ben2(self, image):
        original_size = image.size
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(image)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.numpy(), image, original_size

    def _postprocess_ben2(self, result_np: np.ndarray, im_size: list) -> np.ndarray:
        result = torch.from_numpy(result_np)
        if len(result.shape) == 3:
            result = result.unsqueeze(0)
        result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
        im_array = np.squeeze(im_array)
        return im_array

    def get_ben2_mask(self, image):
        input_data, original_image, (w, h) = self._preprocess_ben2(image)
        input_name = self.ben2_session.get_inputs()[0].name
        outputs = self.ben2_session.run(None, {input_name: input_data})
        alpha = self._postprocess_ben2(outputs[0], im_size=[w, h])
        mask = Image.fromarray(alpha).resize((w, h))
        return mask

    def process_image(self, image: Image.Image):
        """
        Runs the full pipeline: BEN2 for mask, SAM2 for matting.
        Returns:
            - transparent_image: PIL Image with transparent background
        """
        mask_image = self.get_ben2_mask(image).convert("L")
        mask_np = np.array(mask_image)

        raw_mask = (torch.from_numpy(mask_np) / 255.0) > 0
        mask_input = (torch.from_numpy(mask_np) > 0).float() * 20 - 10
        mask_input = mask_input.unsqueeze(0).unsqueeze(0)

        mask_input = torch.nn.functional.interpolate(
            mask_input,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )

        with torch.inference_mode():
            img_prepared = self.sam2_predictor.set_image(image)
            _, alpha, _ = self.sam2_predictor.predict(
                img=img_prepared,
                raw_mask=raw_mask,
                mask_input=mask_input,
                multimask_output=False,
            )

        alpha_result = (alpha * 255).astype(np.uint8).squeeze()
        
        original_image_np = np.array(image.convert("RGB"))
        transparent_image_np = np.concatenate([original_image_np, alpha_result[..., None]], axis=-1)
        transparent_image = Image.fromarray(transparent_image_np, mode="RGBA")
        
        return transparent_image
