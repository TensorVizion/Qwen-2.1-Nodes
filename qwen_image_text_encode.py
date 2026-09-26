"""
Qwen-Image Text Encode
-----------------------
Encodes positive/negative prompts through the Qwen-Image pipeline's own text
encoder. Returns QWEN_IMAGE_COND bundles that QwenImageSampler consumes
directly, since Qwen-Image uses its own encoder rather than ComfyUI's native
CLIP conditioning.
"""

import torch


class QwenImageTextEncode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_image_pipe": ("QWEN_IMAGE_PIPE",),
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("QWEN_IMAGE_COND", "QWEN_IMAGE_COND")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "QwenImage"

    def encode(self, qwen_image_pipe, positive_prompt, negative_prompt):
        pipe = qwen_image_pipe["pipe"]

        with torch.no_grad():
            pos_embeds, pos_mask = pipe.encode_prompt(
                prompt=positive_prompt,
                device=qwen_image_pipe["device"],
            )
            neg_embeds, neg_mask = pipe.encode_prompt(
                prompt=negative_prompt if negative_prompt.strip() else "",
                device=qwen_image_pipe["device"],
            )

        positive = {"embeds": pos_embeds, "mask": pos_mask, "text": positive_prompt}
        negative = {"embeds": neg_embeds, "mask": neg_mask, "text": negative_prompt}
        return (positive, negative)


NODE_CLASS_MAPPINGS = {
    "QwenImageTextEncode": QwenImageTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageTextEncode": "Qwen-Image Text Encode",
}
