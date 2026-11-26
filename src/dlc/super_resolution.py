#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Super-resolution for step d: make a small crop of the rat 4x larger with a
neural network (Real-ESRGAN, model "realesr-general-x4v3").

IMPORTANT, read before using the result for science:
the network does not see more than the camera did. It was trained on many
sharp/blurry picture pairs and fills in detail that LOOKS right (fur texture,
clean edges). That is good for looking at the video. It is not a measurement:
a whisker or an ear edge in the super-resolved video may be drawn slightly off
from where it really was. Step d therefore always also writes the plain
enlarged video, which holds only real pixels.

Needs PyTorch (already installed wherever DeepLabCut 3 runs) and the model
file, about 5 MB. The model file is downloaded once, the first time it is
needed, into <repo>/models/. To use a file you already have, or on a computer
without internet, set DLC_SR_MODEL to its path.
"""
import os
import urllib.request
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

MODEL_FILE_NAME = "realesr-general-x4v3.pth"
MODEL_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
             "v0.2.5.0/" + MODEL_FILE_NAME)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / MODEL_FILE_NAME

# The network always enlarges 4 times; that is how it was trained.
MODEL_SCALE = 4

# Shape of the network (it must match the model file exactly).
NUM_FEATURES = 64       # channels inside the network
NUM_BODY_LAYERS = 32    # convolution layers between the first and the last one


def _build_network():
    """The Real-ESRGAN "compact" network (called SRVGGNetCompact by its authors).

    Built inside a function so that this file can be imported on a computer
    without PyTorch; only load_upscaler() then reports that it is missing."""
    import torch.nn as nn
    import torch.nn.functional as F

    class CompactNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # The layer list must be called "body": the model file stores its
            # numbers under names like "body.0.weight".
            layers = [nn.Conv2d(3, NUM_FEATURES, 3, 1, 1), nn.PReLU(NUM_FEATURES)]
            for _ in range(NUM_BODY_LAYERS):
                layers += [nn.Conv2d(NUM_FEATURES, NUM_FEATURES, 3, 1, 1), nn.PReLU(NUM_FEATURES)]
            # last layer: 16 values (4 x 4) per colour for every input pixel ...
            layers.append(nn.Conv2d(NUM_FEATURES, 3 * MODEL_SCALE * MODEL_SCALE, 3, 1, 1))
            self.body = nn.ModuleList(layers)
            # ... which PixelShuffle lays out as a 4 x 4 block of output pixels
            self.upsampler = nn.PixelShuffle(MODEL_SCALE)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            out = self.upsampler(out)
            # The network only learned what to ADD to a simple enlargement.
            return out + F.interpolate(x, scale_factor=MODEL_SCALE, mode="nearest")

    return CompactNetwork()


def _find_or_download_model(model_path: Optional[str]) -> Path:
    """Path of the model file; downloads it the first time when no path is given."""
    if model_path:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"DLC_SR_MODEL does not exist: {path}")
        return path

    path = DEFAULT_MODEL_PATH
    if path.is_file():
        return path

    print(f"Downloading the super-resolution model (about 5 MB) to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Download under a temporary name and rename at the end, so a download that
    # was cut off never leaves half a file with the real name. The process id in
    # the name keeps two sessions running at once from writing the same file.
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        urllib.request.urlretrieve(MODEL_URL, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return path


class Upscaler:
    """Enlarges small pictures 4x with the network. Use load_upscaler() to get one."""

    def __init__(self, network, device):
        self.network = network
        self.device = device

    def upscale(self, pictures: List[np.ndarray]) -> List[np.ndarray]:
        """`pictures`: same-sized OpenCV pictures (BGR, 0-255). Returns each one
        4x larger, in the same order and the same format."""
        import torch

        # OpenCV: blue-green-red, 0-255, (height, width, colour).
        # Network: red-green-blue, 0-1, (colour, height, width).
        batch = np.stack([cv2.cvtColor(p, cv2.COLOR_BGR2RGB) for p in pictures])
        batch = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255.0

        with torch.inference_mode():
            enlarged = self.network(batch)

        enlarged = (enlarged.clamp(0, 1) * 255.0).round().byte()
        enlarged = enlarged.permute(0, 2, 3, 1).cpu().numpy()
        return [cv2.cvtColor(p, cv2.COLOR_RGB2BGR) for p in enlarged]


def load_upscaler(model_path: Optional[str] = None) -> Optional[Upscaler]:
    """The ready-to-use network, or None (with the reason printed) when it cannot
    be used on this computer. Step d then simply writes no super-resolved video."""
    try:
        import torch
    except ImportError:
        print("Super-resolution skipped: PyTorch is not installed in this Python.")
        return None

    try:
        path = _find_or_download_model(model_path)
        saved = torch.load(str(path), map_location="cpu")
        # the authors keep the weights under "params" (or "params_ema")
        for key in ("params_ema", "params"):
            if isinstance(saved, dict) and key in saved:
                saved = saved[key]
                break
        network = _build_network()
        network.load_state_dict(saved, strict=True)
    except Exception as error:
        print(f"Super-resolution skipped: could not load the model ({error}).")
        return None

    # Graphics card when there is one (NVIDIA on the lab PCs, "mps" on a Mac),
    # else the processor, which works but is much slower.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    network = network.to(device).eval()
    print(f"Super-resolution model loaded ({path.name}, running on {device}).")
    return Upscaler(network, device)
