"""
Package entry point ComfyUI actually loads.

This file is intentionally defensive. Every step is wrapped in its own
try/except, and every status line is written to BOTH stdout (print) and a
plain-text log file next to this __init__.py called _load_debug.log.

Why the log file, not just print: some ComfyUI builds -- the desktop app and
some portable/Windows setups in particular -- don't show a console at all, so
print() output can be genuinely invisible even when everything ran correctly.
The log file is a fallback you can always open directly.

After copying this folder into ComfyUI/custom_nodes/ and restarting ComfyUI,
open:
    ComfyUI/custom_nodes/qwen_image_nodes/_load_debug.log
It is rewritten fresh on every ComfyUI start and will show exactly which of
the 4 node files loaded and, for any that didn't, the full traceback of why.
"""

import os
import sys
import traceback
import importlib
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG_PATH = os.path.join(_HERE, "_load_debug.log")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _log(line=""):
    stamp = datetime.now().strftime("%H:%M:%S")
    msg = f"[QwenImageNodes {stamp}] {line}"
    print(msg)
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass  # printing is still better than a hard crash if disk is read-only


try:
    open(_LOG_PATH, "w", encoding="utf-8").close()  # fresh log every startup
except Exception:
    pass

_log("__init__.py started executing")
_log(f"this file's location: {os.path.abspath(__file__)}")
_log(f"python executable in use: {sys.executable}")
_log(f"python version: {sys.version.split()[0]}")

try:
    import torch
    _log(f"torch import OK (version {torch.__version__})")
except Exception:
    _log("torch import FAILED. This means ComfyUI's own Python environment "
         "is broken, not this node pack specifically:")
    _log(traceback.format_exc())

try:
    import comfy.model_management  # noqa: F401
    _log("comfy.model_management import OK")
except Exception:
    _log("comfy.model_management import FAILED. Most likely cause: this "
         "folder is not sitting directly under ComfyUI/custom_nodes/ "
         "(check for an extra nested qwen_image_nodes/qwen_image_nodes/ "
         "folder from how the zip was extracted):")
    _log(traceback.format_exc())

try:
    import diffusers
    _log(f"diffusers import OK (version {diffusers.__version__})")
except Exception:
    _log("diffusers import FAILED or not installed yet. Run (inside "
         "ComfyUI's own Python environment): pip install -r requirements.txt "
         "-- the 4 nodes will still register below regardless; diffusers is "
         "only needed when you actually run the graph.")

_NODE_MODULES = [
    "qwen_image_loader",
    "qwen_image_text_encode",
    "qwen_image_latent_size",
    "qwen_image_sampler",
]

for _mod_name in _NODE_MODULES:
    try:
        _mod = importlib.import_module(f".{_mod_name}", package=__name__)
        _new_classes = getattr(_mod, "NODE_CLASS_MAPPINGS", {})
        NODE_CLASS_MAPPINGS.update(_new_classes)
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
        _log(f"loaded {_mod_name}.py OK -> registered {list(_new_classes.keys())}")
    except Exception:
        _log(f"FAILED to load {_mod_name}.py -- it will NOT appear in ComfyUI:")
        _log(traceback.format_exc())

_log(f"TOTAL: {len(NODE_CLASS_MAPPINGS)}/{len(_NODE_MODULES)} node(s) registered "
     f"-> {sorted(NODE_CLASS_MAPPINGS.keys())}")

if len(NODE_CLASS_MAPPINGS) == 0:
    _log("Zero nodes registered. If you're reading this in _load_debug.log, "
         "this __init__.py DID execute -- so if ComfyUI's node search still "
         "shows nothing, the issue is elsewhere (browser cache: hard-refresh "
         "with Ctrl+Shift+R, or the frontend simply needs a full restart, "
         "not just a page reload).")

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
