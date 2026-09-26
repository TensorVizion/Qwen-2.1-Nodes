"""
Qwen-Image Sampler
-------------------
Runs the Qwen-Image diffusion transformer's denoising loop over an empty (or
partially noised, for img2img) latent, guided by positive/negative
conditioning from QwenImageTextEncode.
"""

import torch
import comfy.utils


class QwenImageSampler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_image_pipe": ("QWEN_IMAGE_PIPE",),
                "positive": ("QWEN_IMAGE_COND",),
                "negative": ("QWEN_IMAGE_COND",),
                "latent": ("LATENT",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (
                    ["flow_match_euler", "flow_match_heun", "dpmpp_2m"],
                    {"default": "flow_match_euler"},
                ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "QwenImage"

    def sample(self, qwen_image_pipe, positive, negative, latent, steps, cfg, seed, scheduler, denoise):
        pipe = qwen_image_pipe["pipe"]
        device = qwen_image_pipe["device"]

        # Swap in the requested scheduler if the pipeline exposes alternates.
        if hasattr(pipe, "set_scheduler"):
            pipe.set_scheduler(scheduler)

        generator = torch.Generator(device=device).manual_seed(seed)
        samples = latent["samples"].to(device)

        pbar = comfy.utils.ProgressBar(steps)

        def _progress_callback(step, timestep, latents):
            pbar.update(1)

        with torch.no_grad():
            output = pipe(
                prompt_embeds=positive["embeds"],
                prompt_attention_mask=positive["mask"],
                negative_prompt_embeds=negative["embeds"],
                negative_prompt_attention_mask=negative["mask"],
                latents=samples if denoise < 1.0 else None,
                num_inference_steps=steps,
                guidance_scale=cfg,
                strength=denoise,
                generator=generator,
                output_type="latent",
                callback_on_step_end=_progress_callback,
            )

        result_latent = output.images if hasattr(output, "images") else output[0]
        return ({"samples": result_latent.cpu()},)


NODE_CLASS_MAPPINGS = {
    "QwenImageSampler": QwenImageSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageSampler": "Qwen-Image Sampler",
}
