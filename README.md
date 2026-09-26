# ComfyUI Qwen-Image Nodes

Four custom nodes wrapping the Qwen-Image diffusion pipeline for ComfyUI.

> **Naming note:** I couldn't verify a released model called "Qwen 2.1 Image."
> The current publicly documented model in this line is **Qwen-Image**
> (and its edit variant, **Qwen-Image-Edit**), from Alibaba's Qwen team.
> This package targets that pipeline. If a newer point release ships with a
> different repo id or diffusers pipeline class name, update
> `MODEL_REPO_CHOICES` and the `QwenImagePipeline` import at the top of
> `nodes.py`.

## Files

Each node is its own file (one class = one file), and `__init__.py` imports
them individually and merges their mappings:

```
qwen_image_nodes/
├── __init__.py                  # imports the 4 files below, merges mappings, writes a debug log
├── qwen_image_loader.py         # Qwen-Image Loader
├── qwen_image_text_encode.py    # Qwen-Image Text Encode
├── qwen_image_latent_size.py    # Qwen-Image Empty Latent
├── qwen_image_sampler.py        # Qwen-Image Sampler
├── requirements.txt             # diffusers (from GitHub) + supporting libs
└── README.md
```

This loading mechanism (relative imports resolved via
`importlib.import_module(f".{name}", package=__name__)`, each in its own
try/except) has been tested against ComfyUI's actual loader logic
(`importlib.util.spec_from_file_location` pointed at `__init__.py`), not just
run standalone — it correctly registers all 4 nodes end to end.

| Node | Purpose |
|---|---|
| **Qwen-Image Loader** | Loads the DiT transformer + VAE + text encoder from a HF repo or local path. |
| **Qwen-Image Text Encode** | Encodes positive/negative prompts via the model's native text encoder (not ComfyUI's CLIP conditioning). |
| **Qwen-Image Empty Latent** | Builds an empty latent on one of Qwen-Image's native aspect-ratio buckets (or a custom size). |
| **Qwen-Image Sampler** | Runs the denoising loop: steps, CFG, seed, scheduler choice, and img2img-style `denoise` strength. |

## Install

1. Copy the whole `qwen_image_nodes` folder into `ComfyUI/custom_nodes/`, so
   the path looks like `ComfyUI/custom_nodes/qwen_image_nodes/__init__.py`
   (not nested one level deeper — NOT
   `ComfyUI/custom_nodes/qwen_image_nodes/qwen_image_nodes/__init__.py` —
   this is the single most common reason a node pack silently fails to appear).
2. Install dependencies into **ComfyUI's own Python environment** (see the
   comment block at the top of `requirements.txt` for the exact command for
   portable/venv/conda installs):
   ```
   pip install -r requirements.txt
   ```
3. Restart ComfyUI, then check for node registration in **either** of two
   places (use whichever you actually have access to):
   - **Console**, if you have one: a line like
     `[QwenImageNodes 12:34:56] TOTAL: 4/4 node(s) registered -> [...]`
   - **Log file**, always available regardless of console visibility: open
     `ComfyUI/custom_nodes/qwen_image_nodes/_load_debug.log` — it's rewritten
     fresh every ComfyUI startup and lists exactly which of the 4 files loaded,
     plus a full traceback for any that didn't.
4. The nodes appear under the **QwenImage** category in the node search/add menu.

### If `_load_debug.log` doesn't even exist after restarting

That means `__init__.py` itself never ran, which narrows it down to:
- The folder isn't directly under `custom_nodes/` (see the nesting note above).
- ComfyUI is looking at a different `custom_nodes/` folder than you think —
  check the actual ComfyUI install path it printed at startup.
- The folder or a parent folder name starts with `.` or `__` (ComfyUI skips
  those).

### If `_load_debug.log` exists and shows `4/4 node(s) registered` but the nodes still don't show up in the UI

At that point it's registered correctly on the Python side, so it's a
frontend issue, not this node pack — hard-refresh the browser tab
(Ctrl+Shift+R / Cmd+Shift+R) to clear cached node-definition JS, or fully
restart ComfyUI rather than just reloading the page.

## Basic graph

```
Qwen-Image Loader ──┬─────────────────────────────► Qwen-Image Sampler
                     │                                     ▲   ▲
                     └──► Qwen-Image Text Encode ──positive─┘   │
                              (pos + neg prompts) ──negative────┘
Qwen-Image Empty Latent ─────────────────────────────► (latent input)
Qwen-Image Sampler ──► (LATENT output) ──► VAE Decode (use the pipe's VAE
                                              or ComfyUI's standard VAE Decode
                                              node if you route the VAE out
                                              separately)
```

## Known assumptions / things to check against your installed diffusers version

- `QwenImagePipeline.encode_prompt(...)` and the pipeline's `__call__` kwargs
  (`prompt_embeds`, `negative_prompt_embeds`, `output_type="latent"`, etc.)
  are assumed to follow diffusers' standard conventions for this pipeline
  family. Check `python -c "from diffusers import QwenImagePipeline; help(QwenImagePipeline.__call__)"`
  if you hit a `TypeError` on unexpected kwargs, and adjust the call in
  `QwenImageSampler.sample()` accordingly.
- Latent channel count (16) and VAE scale factor (8) in
  `QwenImageLatentSize` match Qwen-Image's published architecture; update
  these constants if a future version changes them.
- `pipe.set_scheduler(...)` is a convenience placeholder — swap in whatever
  diffusers scheduler class the pipeline actually expects
  (e.g. `pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(...)`).
