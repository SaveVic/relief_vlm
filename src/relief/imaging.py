"""
relief.imaging
==============
Two separate image concerns:

1. `encode_image` / `load_condition_images`: prepare images for the model.
   Used by every API call.
2. `build_fusion` / `build_cavity` / `build_both`: enhancement from photo +
   depth map. The dataset already ships `fusion.png` and `cavity.png`, so the
   experiment does not call these; they are kept to regenerate the derived
   images and as the method reference for the paper.

Note on `fast_blur`: the enhancement pipeline needs sigma up to 500 px to
remove wall curvature. On a 4608x2592 image one Gaussian blur that large
takes about 22 seconds. The result is purely low-pass, with no content above
~1/500 cycles per pixel, so it can be computed on a downscaled image and
upsampled with a near-identical result.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np

cv2.setNumThreads(cv2.getNumberOfCPUs())

# Image file names inside each panel folder.
IMAGE_FILES = {
    "RGB": "rgb.png",
    "DEPTH": "depth.png",
    "FUSION": "fusion.png",
    "CAVITY": "cavity.png",
}


# ==========================================================================
# Image preparation for the API
# ==========================================================================

def encode_image(
    src: Union[str, Path, np.ndarray],
    max_side: int = 1536,
    jpeg_quality: int = 90,
) -> bytes:
    """
    Read, resize and compress one image to JPEG bytes.

    Returns raw bytes, not a data URI: how they are sent (Files API upload
    or data URI) is the client's decision, not this function's.
    """
    if isinstance(src, (str, Path)):
        path = Path(src)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path.resolve()}")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"OpenCV failed to read image: {path.resolve()}")
    else:
        img = np.asarray(src)
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG compression failed")
    return buf.tobytes()


def image_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_condition_images(
    root: Union[str, Path], panel_id: str, image_keys: List[str]
) -> List[Tuple[str, Path]]:
    """
    Locate the image files one condition needs in a panel folder.

    Returns [(label, path), ...] in exactly the order of `image_keys`. The
    RGB_X3 condition lists "RGB" three times on purpose: it is the
    image-count control, sending three images without adding information.
    """
    folder = Path(root) / panel_id
    out: List[Tuple[str, Path]] = []
    for key in image_keys:
        path = folder / IMAGE_FILES[key]
        if not path.exists():
            raise FileNotFoundError(path)
        out.append((key, path))
    return out


# ==========================================================================
# Enhancement: photo + depth -> fusion / cavity
# ==========================================================================

def fast_blur(x, sigma, target_sigma=6.0):
    """Gaussian blur whose cost does not explode at large sigma."""
    if sigma <= target_sigma * 1.5:
        return cv2.GaussianBlur(x, (0, 0), sigma)

    h, w = x.shape[:2]
    k = int(min(sigma / target_sigma, min(h, w) / 16))
    if k < 2:
        return cv2.GaussianBlur(x, (0, 0), sigma)

    small = cv2.resize(x, (max(w // k, 8), max(h // k, 8)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / k)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def detrend(x, sigma):
    """Remove low-frequency components: wall curvature, vignetting,
    lighting gradients. Leaves variation at the scale of the carving."""
    return x - fast_blur(x, sigma)


def norm01(x, lo=1.0, hi=99.0, sample=2_000_000):
    """
    Percentile-based contrast stretch, robust to outliers.

    Percentiles come from a strided subsample: on 12 million pixels the
    difference is invisible, and it avoids repeated full sorts.
    """
    flat = x.reshape(-1)
    if flat.size > sample:
        flat = flat[:: flat.size // sample]
    a, b = (float(v) for v in np.percentile(flat, [lo, hi]))
    # a and b are deliberately Python floats, not np.float64: a numpy scalar
    # would promote a float32 array to float64, and cv2.Sobel rejects a
    # float64 source with a CV_32F destination.
    return np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0)


def load_pair(img_path, depth_path, depth_bright_is_far=True):
    """Read the photo (BGR float 0..1) and depth; depth is returned as a
    0..1 height with bright = protruding towards the camera."""
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    dep = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if bgr is None or dep is None:
        raise FileNotFoundError("photo or depth could not be read")

    bgr = bgr.astype(np.float32) / 255.0
    if dep.ndim == 3:
        dep = cv2.cvtColor(dep, cv2.COLOR_BGR2GRAY)
    dep = dep.astype(np.float32)

    if dep.shape[:2] != bgr.shape[:2]:
        dep = cv2.resize(dep, (bgr.shape[1], bgr.shape[0]),
                         interpolation=cv2.INTER_CUBIC)

    span = max(dep.max() - dep.min(), 1e-6)
    height = (dep.max() - dep) / span if depth_bright_is_far else (dep - dep.min()) / span
    return bgr, height


def pseudo_height_from_photo(lum, scale=3, scales=((3, 0.30), (9, 0.35), (28, 0.35))):
    """
    Pseudo-height from photo luminance.

    Built at reduced resolution and upsampled, so the andesite grain is not
    lifted into noise when gradients are taken later.
    """
    h, w = lum.shape
    small = cv2.resize(lum, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    small = cv2.bilateralFilter(small, 13, 0.10, 13)

    acc = np.zeros_like(small)
    for sigma, weight in scales:
        # sigma is already in small-image pixels; do not divide by `scale` again
        acc += weight * detrend(small, sigma)

    acc = cv2.GaussianBlur(norm01(acc), (0, 0), 1.2)
    return cv2.resize(acc, (w, h), interpolation=cv2.INTER_CUBIC)


def hybrid_height(bgr, depth, depth_weight=0.40, depth_blur=60, depth_detrend=500, lum=None):
    """Combine large-scale shape from depth with detail from the photo."""
    if lum is None:
        lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h_low = norm01(detrend(fast_blur(depth, depth_blur), depth_detrend))
    h_photo = pseudo_height_from_photo(lum)
    return np.clip(depth_weight * h_low + (1.0 - depth_weight) * h_photo,
                   0.0, 1.0).astype(np.float32)


def surface_normals(height, strength=6.0):
    """Surface normals from a height field. `strength` exaggerates relief;
    too large a value brings out stone texture rather than carving."""
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(height)
    mag = np.sqrt(nx * nx + ny * ny + nz * nz)
    return nx / mag, ny / mag, nz / mag


def raking_light(height, azimuth=135.0, elevation=18.0, strength=6.0, ambient=0.20):
    """
    Render raking light, mimicking the archaeological photography technique.

    azimuth   : light direction in degrees. 135 = upper left, 45 = upper right.
    elevation : a low angle (10-20) maximises shadows from shallow carving.
    """
    nx, ny, nz = surface_normals(height, strength)
    az, el = np.radians(azimuth), np.radians(elevation)
    lx = np.cos(az) * np.cos(el)
    ly = -np.sin(az) * np.cos(el)
    lz = np.sin(el)
    lam = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    return norm01(ambient + (1.0 - ambient) * lam)


def cavity_map(height, sigmas=(5, 15, 40)):
    """Simple ambient occlusion: how deep each pixel sits relative to its
    surroundings, summed over several scales."""
    acc = np.zeros_like(height)
    for s in sigmas:
        acc += np.clip(fast_blur(height, s) - height, 0.0, None)
    return norm01(acc)


def flatten_lighting(lum, clip_limit=2.2, tiles=20, unsharp_sigma=30, unsharp_amount=0.45):
    """Flatten photo lighting, then boost local contrast at carving scale."""
    lum8 = (norm01(lum, 0.2, 99.8) * 255).astype(np.uint8)
    flat = cv2.createCLAHE(clip_limit, (tiles, tiles)).apply(lum8).astype(np.float32) / 255.0
    return np.clip(flat + unsharp_amount * detrend(flat, unsharp_sigma), 0.0, 1.0)


def build_cavity(bgr, depth, gain=2.2, height=None, cav=None):
    """
    Inverted cavity map: carving becomes dark lines on a light background.
    Easiest to read for iconography (pose, attributes, figure count).
    """
    if cav is None:
        cav = cavity_map(hybrid_height(bgr, depth) if height is None else height)
    return np.clip(1.0 - gain * cav, 0.0, 1.0)


def build_fusion(bgr, depth, azimuth=135.0, elevation=18.0, strength=6.0,
                 shade_amount=0.45, cavity_amount=0.20, detail_amount=0.30,
                 height=None, cav=None, lum=None):
    """
    Original stone texture combined with raking-light shading and a little
    cavity. Keeps the look of a real photo while bringing out the relief.
    """
    if lum is None:
        lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if height is None:
        height = hybrid_height(bgr, depth, lum=lum)
    if cav is None:
        cav = cavity_map(height)

    flat = flatten_lighting(lum)
    shade = raking_light(height, azimuth, elevation, strength)
    ao = np.clip(1.0 - 2.2 * cav, 0.0, 1.0)

    out = (0.35
           + 0.50 * flat
           + shade_amount * (shade - 0.5)
           + cavity_amount * (ao - 0.5))
    out = np.clip(out, 0.0, 1.0)
    out = np.clip(out + detail_amount * detrend(out, 18), 0.0, 1.0)
    return norm01(out, 0.5, 99.5)


def build_both(bgr, depth, gain=2.2, **fusion_kwargs):
    """
    Produce fusion and cavity together.

    The height field and cavity map are computed once, instead of twice as
    when the two functions are called separately.
    """
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height = hybrid_height(bgr, depth, lum=lum)
    cav = cavity_map(height)
    return (build_fusion(bgr, depth, height=height, cav=cav, lum=lum, **fusion_kwargs),
            build_cavity(bgr, depth, gain=gain, cav=cav))


def to_uint8(x):
    """float 0..1 -> uint8, ready for cv2.imwrite."""
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)
