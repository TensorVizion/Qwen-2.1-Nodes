"""
Qwen-Image Loader
-----------------
Loads the Qwen-Image diffusion pipeline (transformer + VAE + text encoder)
and exposes it as a single QWEN_IMAGE_PIPE object for the other nodes in
this package to consume.

diffusers is imported lazily inside load(), not at module import time, so a
missing/outdated diffusers install won't prevent this node from registering
in ComfyUI's node menu -- it will only error when you actually run the graph.
"""

import torch
import comfy.model_management as model_management

MODEL_REPO_CHOICES = [
    "Qwen/Qwen-Image",
    "Qwen/Qwen-Image-Edit",
]


class QwenImageLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_repo": (MODEL_REPO_CHOICES, {"default": MODEL_REPO_CHOICES[0]}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "enable_cpu_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "local_path_override": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("QWEN_IMAGE_PIPE",)
    RETURN_NAMES = ("qwen_image_pipe",)
    FUNCTION = "load"
    CATEGORY = "QwenImage"

    def load(self, model_repo, dtype, enable_cpu_offload, local_path_override=""):
        try:
            # Use the generic DiffusionPipeline factory rather than importing
            # QwenImagePipeline by name: Qwen-Image support is very recent,
            # and `from diffusers import QwenImagePipeline` raises ImportError
            # on diffusers versions installed from PyPI before that pipeline
            # was added to a tagged release. DiffusionPipeline.from_pretrained
            # auto-resolves to the correct pipeline class from the repo's
            # model_index.json regardless, and is what Qwen's own model card
            # recommends: https://huggingface.co/Qwen/Qwen-Image
            from diffusers import DiffusionPipeline
        except ImportError as e:
            raise ImportError(
                "Could not import diffusers at all. Install requirements.txt "
                "into ComfyUI's own Python environment first."
            ) from e

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]

        source = local_path_override.strip() or model_repo
        device = model_management.get_torch_device()

        pipe = DiffusionPipeline.from_pretrained(source, torch_dtype=torch_dtype)

        if enable_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        return ({"pipe": pipe, "device": device, "dtype": torch_dtype},)


NODE_CLASS_MAPPINGS = {
    "QwenImageLoader": QwenImageLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageLoader": "Qwen-Image Loader",
}
