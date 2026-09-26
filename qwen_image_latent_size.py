"""
Qwen-Image Empty Latent
------------------------
Builds an empty latent sized to one of Qwen-Image's native aspect-ratio
buckets (or a custom width/height), pre-divided by the model's VAE scale
factor and channel count.
"""

import torch

# Aspect-ratio buckets Qwen-Image documents as native resolutions.
# Update this table if a newer point release changes recommended sizes.
QWEN_IMAGE_BUCKETS = {
    "1:1  (1328x1328)": (1328, 1328),
    "16:9 (1664x928)": (1664, 928),
    "9:16 (928x1664)": (928, 1664),
    "4:3  (1472x1140)": (1472, 1140),
    "3:4  (1140x1472)": (1140, 1472),
    "3:2  (1584x1056)": (1584, 1056),
    "2:3  (1056x1584)": (1056, 1584),
}


class QwenImageLatentSize:

    LATENT_CHANNELS = 16       # Qwen-Image's VAE latent channel count
    VAE_SCALE_FACTOR = 8       # spatial downscale of the VAE

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bucket": (list(QWEN_IMAGE_BUCKETS.keys()) + ["custom"], {"default": "1:1  (1328x1328)"}),
                "custom_width": ("INT", {"default": 1328, "min": 64, "max": 4096, "step": 16}),
                "custom_height": ("INT", {"default": 1328, "min": 64, "max": 4096, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "build"
    CATEGORY = "QwenImage"

    def build(self, bucket, custom_width, custom_height, batch_size):
        if bucket == "custom":
            width, height = custom_width, custom_height
        else:
            width, height = QWEN_IMAGE_BUCKETS[bucket]

        # round to nearest multiple of the VAE scale factor, just in case
        width = (width // self.VAE_SCALE_FACTOR) * self.VAE_SCALE_FACTOR
        height = (height // self.VAE_SCALE_FACTOR) * self.VAE_SCALE_FACTOR

        latent_w = width // self.VAE_SCALE_FACTOR
        latent_h = height // self.VAE_SCALE_FACTOR

        samples = torch.zeros(
            [batch_size, self.LATENT_CHANNELS, latent_h, latent_w]
        )
        return ({"samples": samples}, width, height)


NODE_CLASS_MAPPINGS = {
    "QwenImageLatentSize": QwenImageLatentSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageLatentSize": "Qwen-Image Empty Latent",
}
