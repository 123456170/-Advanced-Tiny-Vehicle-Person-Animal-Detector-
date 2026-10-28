from __future__ import annotations

import json
import math
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import streamlit as st


# =============================================================================
# Constants
# =============================================================================

CLASS_NAMES: Tuple[str, ...] = ("vehicle", "person", "animal")

CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "vehicle": (0, 255, 255),
    "person": (0, 180, 255),
    "animal": (255, 180, 0),
}

TINY_AREA = 16 * 16
SMALL_AREA = 32 * 32
MEDIUM_AREA = 96 * 96

TARGET_FPS_FOR_BUDGET = 15
TARGET_LATENCY_MS = 1000.0 / TARGET_FPS_FOR_BUDGET

UPLOAD_ROOT = Path(tempfile.gettempdir()) / "uav_real_sprites"

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

DEFAULT_PERSON_SPRITE_KEY = "real:person"
DEFAULT_ANIMAL_SPRITE_KEY = "real:animal"
DEFAULT_VEHICLE_SPRITE_KEY = "real:vehicle"

# Direct real-image URLs. These are attempted first.
REAL_IMAGE_URLS: Dict[str, List[str]] = {
    "person": [
        "https://loremflickr.com/320/240/person,walking",
        "https://loremflickr.com/320/240/man,street",
        "https://loremflickr.com/320/240/woman,street",
    ],
    "animal": [
        "https://loremflickr.com/320/240/dog",
        "https://loremflickr.com/320/240/cat",
        "https://loremflickr.com/320/240/horse",
    ],
    "vehicle": [
        "https://loremflickr.com/320/240/car",
        "https://loremflickr.com/320/240/automobile",
        "https://loremflickr.com/320/240/van",
    ],
}

# Wikimedia Commons search queries used if direct URLs fail.
COMMONS_QUERIES: Dict[str, str] = {
    "person": "pedestrian walking street",
    "animal": "dog outdoors",
    "vehicle": "car road",
}

# Wikipedia article fallbacks used if Commons search fails.
WIKI_TITLES: Dict[str, str] = {
    "person": "Pedestrian|Person|Human",
    "animal": "Dog|Cat|Horse",
    "vehicle": "Car|Automobile|Van",
}


# =============================================================================
# Data models
# =============================================================================


@dataclass
class Config:
    width: int = 1920
    height: int = 1080
    fps: int = 25
    duration_s: int = 8
    object_count: int = 90
    seed: int = 42

    sensitivity: float = 1.00
    threshold: float = 0.30

    use_tiling: bool = True
    tile_size: int = 640
    tile_overlap: float = 0.20

    use_wbf: bool = True
    nms_iou: float = 0.45

    use_tracker: bool = True
    tracker_iou: float = 0.30
    tracker_alpha: float = 0.55
    min_track_hits: int = 2
    max_tracker_misses: int = 5

    calibrate: bool = True

    motion_blur: bool = True
    compression_noise: bool = True
    glare: bool = False
    shadows: bool = True
    gps_overlay: bool = True
    empty_frames: bool = False

    use_real_sprites: bool = True
    person_sprite_key: str = DEFAULT_PERSON_SPRITE_KEY
    animal_sprite_key: str = DEFAULT_ANIMAL_SPRITE_KEY
    vehicle_sprite_key: str = DEFAULT_VEHICLE_SPRITE_KEY


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    cls: str
    conf: float
    track_id: Optional[int] = None
    source: str = "raw"
    smoothed: bool = False


@dataclass
class Track:
    id: int
    cls: str
    bbox: Tuple[int, int, int, int]
    conf: float
    hits: int = 1
    misses: int = 0
    last_frame: int = 0


@dataclass
class SceneObject:
    id: int
    cls: str
    x: float
    y: float
    vx: float
    vy: float
    length: float
    width: float
    color: Tuple[int, int, int]
    prev_x: float
    prev_y: float


# =============================================================================
# Geometry helpers
# =============================================================================


def bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def clip_bbox(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1_i = int(math.floor(max(0.0, min(float(width - 1), x1))))
    x2_i = int(math.ceil(max(0.0, min(float(width - 1), x2))))
    y1_i = int(math.floor(max(0.0, min(float(height - 1), y1))))
    y2_i = int(math.ceil(max(0.0, min(float(height - 1), y2))))

    if x2_i < x1_i:
        x2_i = x1_i
    if y2_i < y1_i:
        y2_i = y1_i

    return x1_i, y1_i, x2_i, y2_i


def iou(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    inter = iw * ih

    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return float(inter / union)


def _rng_choice(rng: np.random.Generator, items: Sequence[str]) -> str:
    return items[int(rng.integers(0, len(items)))]


# =============================================================================
# Post-processing: NMS and WBF
# =============================================================================


def non_max_suppression(
    detections: List[Detection],
    iou_threshold: float,
) -> List[Detection]:
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d.conf, reverse=True)
    kept: List[Detection] = []

    for det in detections:
        dominated = False
        for existing in kept:
            if det.cls == existing.cls and iou(det.bbox, existing.bbox) > iou_threshold:
                dominated = True
                break
        if not dominated:
            kept.append(det)

    return kept


def weighted_box_fusion(
    detections: List[Detection],
    iou_threshold: float = 0.50,
) -> List[Detection]:
    if not detections:
        return []

    by_class: Dict[str, List[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.cls, []).append(det)

    fused: List[Detection] = []

    for cls, group in by_class.items():
        group = sorted(group, key=lambda d: d.conf, reverse=True)
        clusters: List[Dict[str, Any]] = []

        for det in group:
            best_idx = -1
            best_iou = iou_threshold

            for idx, cluster in enumerate(clusters):
                score = iou(det.bbox, cluster["bbox"])
                if score >= best_iou:
                    best_iou = score
                    best_idx = idx

            if best_idx >= 0:
                cluster = clusters[best_idx]
                old_weight = float(cluster["weight"])
                new_weight = float(det.conf)
                total_weight = max(1e-9, old_weight + new_weight)

                ax1, ay1, ax2, ay2 = cluster["bbox"]
                bx1, by1, bx2, by2 = det.bbox

                merged = (
                    int(round((ax1 * old_weight + bx1 * new_weight) / total_weight)),
                    int(round((ay1 * old_weight + by1 * new_weight) / total_weight)),
                    int(round((ax2 * old_weight + bx2 * new_weight) / total_weight)),
                    int(round((ay2 * old_weight + by2 * new_weight) / total_weight)),
                )

                if merged[2] < merged[0]:
                    merged = (merged[2], merged[1], merged[0], merged[3])
                if merged[3] < merged[1]:
                    merged = (merged[0], merged[3], merged[2], merged[1])

                cluster["bbox"] = merged
                cluster["weight"] = total_weight
                cluster["max_conf"] = max(float(cluster["max_conf"]), float(det.conf))
                cluster["count"] = int(cluster["count"]) + 1
            else:
                clusters.append(
                    {
                        "bbox": det.bbox,
                        "weight": float(det.conf),
                        "max_conf": float(det.conf),
                        "count": 1,
                    }
                )

        for cluster in clusters:
            fused.append(
                Detection(
                    bbox=cluster["bbox"],
                    cls=cls,
                    conf=float(cluster["max_conf"]),
                    source="wbf",
                )
            )

    return fused


# =============================================================================
# Tiling
# =============================================================================


def make_tiles(
    width: int,
    height: int,
    tile_size: int,
    overlap_ratio: float,
) -> List[Tuple[int, int, int, int]]:
    if tile_size <= 0:
        return [(0, 0, width, height)]

    if tile_size >= width and tile_size >= height:
        return [(0, 0, width, height)]

    overlap = int(max(0.0, min(0.45, overlap_ratio)) * tile_size)
    stride = max(1, tile_size - overlap)

    tiles: List[Tuple[int, int, int, int]] = []

    y = 0
    while y < height:
        y2 = min(y + tile_size, height)

        x = 0
        while x < width:
            x2 = min(x + tile_size, width)

            if x2 - x >= 32 and y2 - y >= 32:
                tiles.append((x, y, x2, y2))

            if x2 >= width:
                break

            x += stride

        if y2 >= height:
            break

        y += stride

    if not tiles:
        tiles = [(0, 0, width, height)]

    return tiles


# =============================================================================
# Real image loading
# =============================================================================


def _http_get(
    url: str,
    timeout: int = 8,
    max_bytes: int = 5_000_000,
) -> Optional[bytes]:
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "TinyObjectUAVDemo/1.0 "
                    "(local educational demo; no API key)"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                return None
            return data
    except Exception:
        return None


def _prepare_sprite(img: Optional[np.ndarray], max_side: int = 160) -> Optional[np.ndarray]:
    if img is None:
        return None

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return None

    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    img = img[y0:y0 + side, x0:x0 + side]

    if side > max_side:
        img = cv2.resize(
            img,
            (max_side, max_side),
            interpolation=cv2.INTER_AREA,
        )

    return np.ascontiguousarray(img)


@st.cache_data(show_spinner=False)
def load_sprite_from_url(url: str) -> Optional[np.ndarray]:
    data = _http_get(url)
    if data is None:
        return None

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    return _prepare_sprite(img)


@st.cache_data(show_spinner=False)
def load_sprite_from_path(path: str) -> Optional[np.ndarray]:
    if not path:
        return None

    p = Path(path)
    if not p.exists():
        return None

    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return _prepare_sprite(img)


@st.cache_data(show_spinner=False)
def load_commons_sprite(query: str) -> Optional[np.ndarray]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",
        "gsrlimit": "20",
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "iiurlwidth": "320",
    }

    url = COMMONS_API + "?" + urllib.parse.urlencode(params)
    raw = _http_get(url)
    if raw is None:
        return None

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return None

    pages = (payload.get("query") or {}).get("pages") or {}

    for page in sorted(pages.values(), key=lambda p: int(p.get("index", 9999))):
        imageinfo = page.get("imageinfo") or []
        if not imageinfo:
            continue

        info = imageinfo[0]
        mime = str(info.get("mime", ""))
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            continue

        w = int(info.get("width") or 0)
        h = int(info.get("height") or 0)
        if w < 80 or h < 80:
            continue

        aspect = float(w) / float(max(1, h))
        if aspect < 0.25 or aspect > 4.0:
            continue

        img_url = str(info.get("thumburl") or info.get("url") or "")
        if not img_url:
            continue

        img = load_sprite_from_url(img_url)
        if img is not None:
            return img

    return None


@st.cache_data(show_spinner=False)
def load_wikipedia_sprite(title_chain: str) -> Optional[np.ndarray]:
    titles = [t.strip() for t in title_chain.split("|") if t.strip()]

    for title in titles:
        try:
            api_url = (
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + urllib.parse.quote(title)
            )
            raw = _http_get(api_url)
            if raw is None:
                continue

            payload = json.loads(raw.decode("utf-8"))

            thumb = (payload.get("thumbnail") or {}).get("source")
            original = (payload.get("originalimage") or {}).get("source")
            image_url = thumb or original

            if not image_url:
                continue

            img = load_sprite_from_url(str(image_url))
            if img is not None:
                return img
        except Exception:
            continue

    return None


def load_real_sprite(cls: str) -> Optional[np.ndarray]:
    """
    Load a real image for a class.

    Order:
      1. direct real-image URLs
      2. Wikimedia Commons search
      3. Wikipedia article thumbnail
    """
    for url in REAL_IMAGE_URLS.get(cls, []):
        img = load_sprite_from_url(url)
        if img is not None:
            return img

    query = COMMONS_QUERIES.get(cls)
    if query:
        img = load_commons_sprite(query)
        if img is not None:
            return img

    titles = WIKI_TITLES.get(cls)
    if titles:
        img = load_wikipedia_sprite(titles)
        if img is not None:
            return img

    return None


def resolve_sprite(key: str) -> Optional[np.ndarray]:
    if not key:
        return None

    if key.startswith("real:"):
        return load_real_sprite(key[5:])

    if key.startswith("commons:"):
        return load_commons_sprite(key[8:])

    if key.startswith("wiki:"):
        return load_wikipedia_sprite(key[5:])

    if key.startswith("file:"):
        return load_sprite_from_path(key[5:])

    if key.startswith(("http://", "https://")):
        return load_sprite_from_url(key)

    return None


def _save_upload_to_temp(uploaded_file: Any, prefix: str) -> Optional[str]:
    if uploaded_file is None:
        return None

    try:
        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

        data = uploaded_file.getvalue()
        digest = str(hash(data))[:12]

        suffix = Path(str(getattr(uploaded_file, "name", ""))).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            suffix = ".jpg"

        path = UPLOAD_ROOT / f"{prefix}_{digest}{suffix}"
        if not path.exists():
            path.write_bytes(data)

        return str(path)
    except Exception:
        return None


def paste_sprite(
    frame: np.ndarray,
    sprite: Optional[np.ndarray],
    bbox: Tuple[int, int, int, int],
    opacity: float = 0.97,
) -> None:
    if sprite is None:
        return

    x1, y1, x2, y2 = bbox
    w = x2 - x1 + 1
    h = y2 - y1 + 1

    if w <= 0 or h <= 0:
        return

    if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
        return

    interp = cv2.INTER_AREA if (w < sprite.shape[1] or h < sprite.shape[0]) else cv2.INTER_LINEAR

    if sprite.ndim == 3 and sprite.shape[2] == 4:
        rgba = cv2.resize(sprite, (w, h), interpolation=interp)
        rgb = rgba[:, :, :3]
        alpha_channel = rgba[:, :, 3]

        if float(alpha_channel.min()) < 250.0:
            alpha = alpha_channel.astype(np.float32)[:, :, None] / 255.0 * float(opacity)
        else:
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(
                mask,
                (w // 2, h // 2),
                (max(1, w // 2), max(1, h // 2)),
                0,
                0,
                360,
                1.0,
                -1,
            )
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            alpha = mask[:, :, None] * float(opacity)
    else:
        rgb = cv2.resize(sprite[:, :, :3], (w, h), interpolation=interp)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(
            mask,
            (w // 2, h // 2),
            (max(1, w // 2), max(1, h // 2)),
            0,
            0,
            360,
            1.0,
            -1,
        )
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        alpha = mask[:, :, None] * float(opacity)

    roi = frame[y1:y2 + 1, x1:x2 + 1].astype(np.float32)
    blended = roi * (1.0 - alpha) + rgb.astype(np.float32) * alpha
    frame[y1:y2 + 1, x1:x2 + 1] = np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _display_image(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =============================================================================
# Synthetic aerial scene with real sprites
# =============================================================================


class SyntheticAerialScene:
    def __init__(self, cfg: Config, sprites: Dict[str, Optional[np.ndarray]]) -> None:
        self.cfg = cfg
        self.sprites = sprites
        self.width = cfg.width
        self.height = cfg.height
        self.fps = max(1, int(cfg.fps))

        self.detail_scale = max(0.65, min(2.2, (cfg.width / 1920.0) ** 0.75))
        self.object_scale = max(0.85, min(1.45, (cfg.width / 1920.0) ** 0.35))

        self.rng = np.random.default_rng(cfg.seed)
        self.background = self._make_background()
        self.objects = self._make_objects(cfg.object_count)
        self.frame_index = 0

    def _pick_color(self, colors: Sequence[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        return colors[int(self.rng.integers(0, len(colors)))]

    def _make_background(self) -> np.ndarray:
        h, w = self.height, self.width
        base = np.zeros((h, w, 3), dtype=np.uint8)

        for yy in range(h):
            t = yy / max(1, h - 1)
            blue = int(88 - 28 * t)
            green = int(112 - 38 * t)
            red = int(82 + 18 * t)
            base[yy, :, :] = (blue, green, red)

        small_h = max(1, h // 8)
        small_w = max(1, w // 8)
        noise = self.rng.normal(0.0, 9.0, (small_h, small_w, 3)).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        base = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        patch_layer = base.copy()
        patch_count = max(12, int(45 * self.detail_scale))

        for _ in range(patch_count):
            cx = int(self.rng.integers(0, w))
            cy = int(self.rng.integers(0, h))
            ax = int(max(8, self.rng.integers(35, 210) * self.detail_scale))
            ay = int(max(6, self.rng.integers(25, 155) * self.detail_scale))
            angle = float(self.rng.uniform(0.0, 180.0))

            center_color = base[cy, cx].astype(np.float32)
            color_arr = np.clip(
                center_color + self.rng.normal(0.0, 24.0, 3),
                0,
                255,
            ).astype(np.uint8)

            color = (int(color_arr[0]), int(color_arr[1]), int(color_arr[2]))

            cv2.ellipse(
                patch_layer,
                (cx, cy),
                (ax, ay),
                angle,
                0,
                360,
                color,
                -1,
                cv2.LINE_AA,
            )

        base = cv2.addWeighted(base, 0.62, patch_layer, 0.38, 0.0)

        road_layer = base.copy()
        road_count = max(2, int(3 * self.detail_scale))

        for _ in range(road_count):
            xs = np.sort(self.rng.uniform(0, w, 7)).astype(np.int32)
            ys = np.clip(
                np.cumsum(self.rng.normal(h * 0.08, h * 0.07, 7)),
                0,
                h - 1,
            ).astype(np.int32)

            pts = np.column_stack((xs, ys)).reshape(-1, 1, 2)
            road_w = int(max(5, 13 * self.detail_scale))

            cv2.polylines(road_layer, [pts], False, (92, 92, 98), road_w, cv2.LINE_AA)
            cv2.polylines(
                road_layer,
                [pts],
                False,
                (175, 175, 175),
                max(1, road_w // 7),
                cv2.LINE_AA,
            )

        base = road_layer

        river_layer = base.copy()
        xs = np.sort(self.rng.uniform(0, w, 9)).astype(np.int32)
        ys = np.clip(
            np.linspace(0, h - 1, 9) + self.rng.normal(0, h * 0.045, 9),
            0,
            h - 1,
        ).astype(np.int32)

        pts = np.column_stack((xs, ys)).reshape(-1, 1, 2)

        cv2.polylines(
            river_layer,
            [pts],
            False,
            (175, 125, 70),
            int(max(7, 18 * self.detail_scale)),
            cv2.LINE_AA,
        )

        base = cv2.addWeighted(base, 0.76, river_layer, 0.24, 0.0)

        building_count = max(3, int(10 * self.detail_scale))

        for _ in range(building_count):
            bx = int(self.rng.integers(0, max(1, w - 80)))
            by = int(self.rng.integers(0, max(1, h - 80)))
            bw = int(max(10, self.rng.integers(18, 70) * self.detail_scale))
            bh = int(max(10, self.rng.integers(18, 70) * self.detail_scale))

            gray = int(self.rng.integers(95, 175))
            color = (gray, gray, max(60, gray - 12))

            cv2.rectangle(base, (bx, by), (bx + bw, by + bh), color, -1, cv2.LINE_AA)
            cv2.rectangle(base, (bx, by), (bx + bw, by + bh), (45, 45, 45), 1, cv2.LINE_AA)
            cv2.line(
                base,
                (bx, by + bh // 2),
                (bx + bw, by + bh // 2),
                (55, 55, 55),
                1,
                cv2.LINE_AA,
            )

        line_count = max(10, int(28 * self.detail_scale))

        for _ in range(line_count):
            y0 = int(self.rng.integers(0, h))
            y1 = int(np.clip(y0 + self.rng.normal(0, 22 * self.detail_scale), 0, h - 1))
            cv2.line(base, (0, y0), (w - 1, y1), (72, 92, 72), 1, cv2.LINE_AA)

        base = cv2.GaussianBlur(base, (3, 3), 0)
        return base

    def _make_objects(self, count: int) -> List[SceneObject]:
        objects: List[SceneObject] = []

        vehicle_colors = [
            (45, 45, 50),
            (185, 185, 190),
            (35, 35, 160),
            (85, 85, 95),
            (55, 140, 210),
        ]

        person_colors = [
            (0, 0, 205),
            (0, 170, 225),
            (245, 185, 80),
            (75, 75, 75),
            (210, 210, 210),
        ]

        animal_colors = [
            (35, 35, 125),
            (85, 85, 165),
            (205, 205, 225),
            (55, 55, 75),
            (120, 90, 60),
        ]

        fps_motion_scale = self.fps / 25.0

        for oid in range(count):
            u = float(self.rng.random())

            if u < 0.45:
                cls = "vehicle"
            elif u < 0.78:
                cls = "person"
            else:
                cls = "animal"

            if cls == "vehicle":
                length = float(self.rng.uniform(10.0, 24.0) * self.object_scale)
                width = float(self.rng.uniform(4.0, 9.0) * self.object_scale)
                speed = float(self.rng.uniform(0.12, 0.42) * self.object_scale * fps_motion_scale)
                color = self._pick_color(vehicle_colors)

            elif cls == "person":
                length = float(self.rng.uniform(3.0, 6.5) * self.object_scale)
                width = length
                speed = float(self.rng.uniform(0.03, 0.16) * self.object_scale * fps_motion_scale)
                color = self._pick_color(person_colors)

            else:
                length = float(self.rng.uniform(4.5, 12.0) * self.object_scale)
                width = float(self.rng.uniform(3.0, 8.0) * self.object_scale)
                speed = float(self.rng.uniform(0.06, 0.28) * self.object_scale * fps_motion_scale)
                color = self._pick_color(animal_colors)

            angle = float(self.rng.uniform(0.0, 2.0 * math.pi))
            x = float(self.rng.uniform(0.0, float(self.width)))
            y = float(self.rng.uniform(0.0, float(self.height)))

            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed

            objects.append(
                SceneObject(
                    id=oid,
                    cls=cls,
                    x=x,
                    y=y,
                    vx=vx,
                    vy=vy,
                    length=length,
                    width=width,
                    color=color,
                    prev_x=x,
                    prev_y=y,
                )
            )

        return objects

    def _update_object(self, obj: SceneObject) -> None:
        obj.prev_x, obj.prev_y = obj.x, obj.y

        if obj.cls == "vehicle":
            obj.x += obj.vx
            obj.y += obj.vy

        elif obj.cls == "person":
            obj.vx += float(self.rng.normal(0.0, 0.008 * self.object_scale))
            obj.vy += float(self.rng.normal(0.0, 0.008 * self.object_scale))

            sp = math.hypot(obj.vx, obj.vy)
            max_sp = 0.24 * self.object_scale * (self.fps / 25.0)

            if sp > max_sp and sp > 0:
                obj.vx *= max_sp / sp
                obj.vy *= max_sp / sp

            obj.x += obj.vx
            obj.y += obj.vy

        else:
            if self.rng.random() < 0.04:
                angle = float(self.rng.uniform(0.0, 2.0 * math.pi))
                speed = float(
                    self.rng.uniform(0.06, 0.28)
                    * self.object_scale
                    * (self.fps / 25.0)
                )
                obj.vx = math.cos(angle) * speed
                obj.vy = math.sin(angle) * speed

            obj.x += obj.vx
            obj.y += obj.vy

        margin = max(obj.length, obj.width) * 2.5

        if obj.x < -margin:
            obj.x = self.width + margin
        elif obj.x > self.width + margin:
            obj.x = -margin

        if obj.y < -margin:
            obj.y = self.height + margin
        elif obj.y > self.height + margin:
            obj.y = -margin

    def _object_bbox(self, obj: SceneObject) -> Tuple[float, float, float, float]:
        if obj.cls == "vehicle":
            angle = math.degrees(math.atan2(obj.vy, obj.vx))
            rect = (
                (obj.x, obj.y),
                (float(max(2.0, obj.length)), float(max(2.0, obj.width))),
                angle,
            )
            pts = cv2.boxPoints(rect)
            x1 = float(pts[:, 0].min())
            y1 = float(pts[:, 1].min())
            x2 = float(pts[:, 0].max())
            y2 = float(pts[:, 1].max())
            return x1, y1, x2, y2

        half_l = max(1.0, obj.length * 0.5)
        half_w = max(1.0, obj.width * 0.5)
        return obj.x - half_l, obj.y - half_w, obj.x + half_l, obj.y + half_w

    def _draw_shadow(self, frame: np.ndarray, obj: SceneObject) -> None:
        if not self.cfg.shadows:
            return

        dx = int(5 * self.detail_scale)
        dy = int(7 * self.detail_scale)

        sx = int(round(obj.x + dx))
        sy = int(round(obj.y + dy))

        if not (0 <= sx < self.width and 0 <= sy < self.height):
            return

        base_pixel = frame[sy, sx].astype(np.float32)
        shadow = tuple(int(max(0.0, c * 0.42)) for c in base_pixel)

        angle = math.degrees(math.atan2(obj.vy, obj.vx))
        axes = (
            max(1, int(obj.length * 0.55)),
            max(1, int(obj.width * 0.75)),
        )

        cv2.ellipse(
            frame,
            (sx, sy),
            axes,
            angle,
            0,
            360,
            shadow,
            -1,
            cv2.LINE_AA,
        )

    def _draw_object(
        self,
        frame: np.ndarray,
        obj: SceneObject,
        bbox: Tuple[int, int, int, int],
    ) -> None:
        x = int(round(obj.x))
        y = int(round(obj.y))

        if self.cfg.motion_blur:
            thickness = max(1, int(obj.width * 0.55))
            cv2.line(
                frame,
                (int(round(obj.prev_x)), int(round(obj.prev_y))),
                (x, y),
                obj.color,
                thickness,
                cv2.LINE_AA,
            )

        self._draw_shadow(frame, obj)

        sprite = self.sprites.get(obj.cls) if self.cfg.use_real_sprites else None

        if sprite is not None:
            paste_sprite(frame, sprite, bbox, opacity=0.97)
            return

        if obj.cls == "vehicle":
            angle = math.degrees(math.atan2(obj.vy, obj.vx))
            rect = (
                (x, y),
                (int(max(2, obj.length)), int(max(2, obj.width))),
                angle,
            )
            pts = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(frame, [pts], obj.color)
            cv2.polylines(frame, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)

        elif obj.cls == "person":
            radius = max(1, int(obj.length * 0.5))
            cv2.circle(frame, (x, y), radius, obj.color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius, (0, 0, 0), 1, cv2.LINE_AA)

        else:
            angle = math.degrees(math.atan2(obj.vy, obj.vx))
            axes = (
                max(1, int(obj.length * 0.5)),
                max(1, int(obj.width * 0.5)),
            )
            cv2.ellipse(
                frame,
                (x, y),
                axes,
                angle,
                0,
                360,
                obj.color,
                -1,
                cv2.LINE_AA,
            )

    def _draw_gps_overlay(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]

        top_band = int(h * 0.075)
        bottom_band = int(h * 0.075)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, top_band), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - bottom_band), (w, h), (0, 0, 0), -1)

        frame[:] = cv2.addWeighted(frame, 0.62, overlay, 0.38, 0.0)

        font_scale = max(0.35, min(0.90, w / 2400.0))
        thickness = max(1, int(round(font_scale * 1.8)))

        top_text = "UAV-01  ALT 120m  LAT 35.689  LON 139.762  BAT 78%  SATS 14"
        bottom_text = "MODE AUTO  WIND 4.2 m/s  TEMP 18C  LINK OK"

        cv2.putText(
            frame,
            top_text,
            (10, int(top_band * 0.65)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            bottom_text,
            (10, h - int(bottom_band * 0.35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (220, 220, 220),
            thickness,
            cv2.LINE_AA,
        )

    def step(self, cfg: Config) -> Tuple[np.ndarray, List[Tuple[Tuple[int, int, int, int], str]], bool]:
        frame = self.background.copy()

        empty_frame = (
            cfg.empty_frames
            and self.frame_index > 0
            and self.frame_index % 97 == 0
        )

        if empty_frame:
            frame[:] = 0
            if cfg.gps_overlay:
                self._draw_gps_overlay(frame)
            self.frame_index += 1
            return frame, [], True

        illum = (
            0.93
            + 0.07 * math.sin(self.frame_index / 28.0)
            + float(self.rng.normal(0.0, 0.003))
        )
        frame = np.clip(frame.astype(np.float32) * illum, 0.0, 255.0).astype(np.uint8)

        gt_boxes: List[Tuple[Tuple[int, int, int, int], str]] = []

        for obj in self.objects:
            self._update_object(obj)

            raw_bbox = self._object_bbox(obj)
            bbox = clip_bbox(raw_bbox, self.width, self.height)

            if bbox_area(bbox) >= 1:
                gt_boxes.append((bbox, obj.cls))

            self._draw_object(frame, obj, bbox)

        if cfg.glare:
            glare_layer = frame.copy()
            center = (int(self.width * 0.76), int(self.height * 0.18))
            radius = int(min(self.width, self.height) * 0.34)
            cv2.circle(glare_layer, center, radius, (255, 255, 255), -1, cv2.LINE_AA)
            frame = cv2.addWeighted(frame, 0.78, glare_layer, 0.22, 0.0)

        if cfg.compression_noise:
            small_h = max(1, self.height // 4)
            small_w = max(1, self.width // 4)

            noise = self.rng.normal(0.0, 1.6, (small_h, small_w, 3)).astype(np.float32)
            noise = cv2.resize(noise, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            frame = np.clip(frame.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

            half_w = max(1, self.width // 2)
            half_h = max(1, self.height // 2)
            small = cv2.resize(frame, (half_w, half_h), interpolation=cv2.INTER_AREA)
            frame = cv2.resize(small, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        if cfg.gps_overlay:
            self._draw_gps_overlay(frame)

        self.frame_index += 1
        return frame, gt_boxes, False


# =============================================================================
# Simulated tiny-object detector
# =============================================================================


def _object_detection(
    bbox: Tuple[int, int, int, int],
    cls: str,
    rng: np.random.Generator,
    cfg: Config,
    tile_size: Optional[Tuple[int, int]] = None,
) -> Optional[Detection]:
    x1, y1, x2, y2 = bbox

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    area = bw * bh
    size = math.sqrt(area)

    class_priors = {
        "vehicle": 0.88,
        "person": 0.78,
        "animal": 0.72,
    }

    prior = class_priors.get(cls, 0.70)
    size_factor = min(1.0, max(0.16, size / 24.0))

    conf = prior * (0.40 + 0.60 * size_factor) * cfg.sensitivity
    conf += float(rng.normal(0.0, 0.035))

    if tile_size is not None:
        tw, th = tile_size
        edge_margin = max(2, int(0.04 * min(tw, th)))
        edge_min = min(x1, y1, tw - 1 - x2, th - 1 - y2)

        if edge_min < edge_margin:
            conf *= 0.86

    if rng.random() < 0.03:
        conf *= 0.72

    conf = float(np.clip(conf, 0.02, 0.99))

    detect_prob = float(np.clip(conf + 0.10, 0.04, 0.985))
    if rng.random() > detect_prob:
        return None

    jitter = max(0.35, 0.55 * (14.0 / max(size, 3.0)))

    nx1 = x1 + rng.normal(0.0, jitter)
    ny1 = y1 + rng.normal(0.0, jitter)
    nx2 = x2 + rng.normal(0.0, jitter)
    ny2 = y2 + rng.normal(0.0, jitter)

    bx1, bx2 = sorted((nx1, nx2))
    by1, by2 = sorted((ny1, ny2))

    if tile_size is not None:
        tw, th = tile_size
        bx1 = float(np.clip(round(bx1), 0, tw - 1))
        bx2 = float(np.clip(round(bx2), 0, tw - 1))
        by1 = float(np.clip(round(by1), 0, th - 1))
        by2 = float(np.clip(round(by2), 0, th - 1))
    else:
        bx1 = float(np.clip(round(bx1), 0, cfg.width - 1))
        bx2 = float(np.clip(round(bx2), 0, cfg.width - 1))
        by1 = float(np.clip(round(by1), 0, cfg.height - 1))
        by2 = float(np.clip(round(by2), 0, cfg.height - 1))

    if bx2 < bx1:
        bx1, bx2 = bx2, bx1
    if by2 < by1:
        by1, by2 = by2, by1

    out_cls = cls

    if size < 12.0 and rng.random() < 0.035:
        others = [c for c in CLASS_NAMES if c != cls]
        out_cls = _rng_choice(rng, others)

    return Detection(
        bbox=(int(bx1), int(by1), int(bx2), int(by2)),
        cls=out_cls,
        conf=conf,
        source="proposal",
    )


def simulate_tile_detections(
    tile_img: np.ndarray,
    tile_box: Tuple[int, int, int, int],
    gt_boxes: Sequence[Tuple[Tuple[int, int, int, int], str]],
    rng: np.random.Generator,
    cfg: Config,
) -> List[Detection]:
    tx1, ty1, tx2, ty2 = tile_box
    tw = tx2 - tx1
    th = ty2 - ty1

    if tw <= 0 or th <= 0 or tile_img.size == 0:
        return []

    local_gts: List[Tuple[Tuple[int, int, int, int], str]] = []

    for bbox, cls in gt_boxes:
        gx1, gy1, gx2, gy2 = bbox

        ix1 = max(gx1, tx1)
        iy1 = max(gy1, ty1)
        ix2 = min(gx2, tx2 - 1)
        iy2 = min(gy2, ty2 - 1)

        if ix2 < ix1 or iy2 < iy1:
            continue

        inter = (ix2 - ix1 + 1) * (iy2 - iy1 + 1)
        orig = bbox_area(bbox)

        if orig > 0 and inter / float(orig) < 0.20:
            continue

        local = (ix1 - tx1, iy1 - ty1, ix2 - tx1, iy2 - ty1)
        local_gts.append((local, cls))

    detections: List[Detection] = []

    for local_bbox, cls in local_gts:
        det = _object_detection(
            bbox=local_bbox,
            cls=cls,
            rng=rng,
            cfg=cfg,
            tile_size=(tw, th),
        )

        if det is not None:
            gx1 = det.bbox[0] + tx1
            gy1 = det.bbox[1] + ty1
            gx2 = det.bbox[2] + tx1
            gy2 = det.bbox[3] + ty1
            det.bbox = (gx1, gy1, gx2, gy2)
            detections.append(det)

    fp_lambda = 0.03 + 0.12 * max(0.0, 1.0 - cfg.sensitivity)

    gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
    texture = float(np.std(gray)) / 255.0
    fp_lambda += 0.08 * texture

    if cfg.compression_noise:
        fp_lambda += 0.04

    if cfg.glare:
        fp_lambda += 0.05

    if cfg.gps_overlay:
        top_band = int(cfg.height * 0.075)
        bottom_start = int(cfg.height * (1.0 - 0.075))

        if ty1 < top_band or ty2 > bottom_start:
            fp_lambda += 0.10

    num_fp = int(rng.poisson(max(0.0, fp_lambda)))

    for _ in range(num_fp):
        s = int(rng.integers(3, 14))
        x = int(rng.integers(0, max(1, tw - s - 1)))
        y = int(rng.integers(0, max(1, th - s - 1)))

        y2 = int(np.clip(y + int(s * rng.uniform(0.7, 1.3)), 0, th - 1))
        if y2 < y:
            y, y2 = y2, y

        conf = float(np.clip(rng.uniform(0.12, 0.42) * cfg.sensitivity, 0.02, 0.92))
        cls = _rng_choice(rng, CLASS_NAMES)

        detections.append(
            Detection(
                bbox=(x + tx1, y + ty1, x + tx1 + s, y2 + ty1),
                cls=cls,
                conf=conf,
                source="fp",
            )
        )

    return detections


# =============================================================================
# Thresholding, calibration, OSD suppression
# =============================================================================


def apply_threshold_and_calibration(
    detections: List[Detection],
    cfg: Config,
) -> List[Detection]:
    class_factor = {
        "vehicle": 0.92,
        "person": 1.00,
        "animal": 1.08,
    }

    kept: List[Detection] = []

    for det in detections:
        conf = float(det.conf)

        if cfg.calibrate:
            sigmoid = 1.0 / (1.0 + math.exp(-8.0 * (conf - 0.35)))
            conf = float(np.clip(0.75 * conf + 0.25 * sigmoid, 0.0, 0.99))

        thr = float(cfg.threshold * class_factor.get(det.cls, 1.0))

        if conf >= thr:
            kept.append(
                Detection(
                    bbox=det.bbox,
                    cls=det.cls,
                    conf=conf,
                    track_id=det.track_id,
                    source=det.source,
                    smoothed=det.smoothed,
                )
            )

    return kept


def suppress_osd_false_alarms(
    detections: List[Detection],
    cfg: Config,
    width: int,
    height: int,
    events: List[Dict[str, Any]],
    frame_idx: int,
) -> List[Detection]:
    if not cfg.gps_overlay:
        return detections

    top_band = int(height * 0.085)
    bottom_band = int(height * 0.915)

    kept: List[Detection] = []

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cy = 0.5 * (y1 + y2)
        area = bbox_area(det.bbox)

        in_osd = cy < top_band or cy > bottom_band

        if in_osd and area < SMALL_AREA and det.conf < 0.68:
            events.append(
                {
                    "frame": int(frame_idx),
                    "type": "osd_suppression",
                    "severity": "info",
                    "message": "Suppressed low-confidence tiny detection in GPS/OSD band",
                    "class": det.cls,
                    "confidence": round(float(det.conf), 3),
                    "bbox": [int(v) for v in det.bbox],
                }
            )
            continue

        kept.append(det)

    return kept


# =============================================================================
# Lightweight tracker
# =============================================================================


class TinyTracker:
    def __init__(
        self,
        iou_threshold: float,
        max_misses: int,
        alpha: float,
        min_hits: int,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_misses = int(max_misses)
        self.alpha = float(alpha)
        self.min_hits = int(min_hits)
        self.tracks: List[Track] = []
        self.next_id: int = 1

    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
    ) -> List[Detection]:
        if not detections:
            for tr in self.tracks:
                tr.misses += 1
            self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]
            return []

        initial_ids = {tr.id for tr in self.tracks}
        active = {tr.id: tr for tr in self.tracks}
        used_ids: set[int] = set()

        det_items = sorted(enumerate(detections), key=lambda p: p[1].conf, reverse=True)
        output: List[Detection] = []

        for _, det in det_items:
            best_id: Optional[int] = None
            best_iou = self.iou_threshold

            for tr in self.tracks:
                if tr.id in used_ids or tr.cls != det.cls:
                    continue

                score = iou(det.bbox, tr.bbox)
                if score >= best_iou:
                    best_iou = score
                    best_id = tr.id

            if best_id is not None:
                tr = active[best_id]
                used_ids.add(best_id)

                ax1, ay1, ax2, ay2 = tr.bbox
                bx1, by1, bx2, by2 = det.bbox

                smoothed = (
                    int(round(self.alpha * bx1 + (1.0 - self.alpha) * ax1)),
                    int(round(self.alpha * by1 + (1.0 - self.alpha) * ay1)),
                    int(round(self.alpha * bx2 + (1.0 - self.alpha) * ax2)),
                    int(round(self.alpha * by2 + (1.0 - self.alpha) * ay2)),
                )

                tr.bbox = smoothed
                tr.conf = float(np.clip(0.65 * det.conf + 0.35 * tr.conf, 0.0, 0.99))
                tr.hits += 1
                tr.misses = 0
                tr.last_frame = frame_idx

                det.bbox = tr.bbox
                det.track_id = tr.id
                det.conf = tr.conf
                det.smoothed = True

                output.append(det)

            else:
                tr = Track(
                    id=self.next_id,
                    cls=det.cls,
                    bbox=det.bbox,
                    conf=det.conf,
                    hits=1,
                    misses=0,
                    last_frame=frame_idx,
                )

                self.tracks.append(tr)
                active[tr.id] = tr
                self.next_id += 1

                det.track_id = tr.id

                if self.min_hits > 1:
                    det.conf = float(np.clip(det.conf * 0.78, 0.0, 0.99))

                output.append(det)

        for tr in self.tracks:
            if tr.id in initial_ids and tr.id not in used_ids:
                tr.misses += 1

        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]

        return output


# =============================================================================
# Drawing helpers
# =============================================================================


def draw_tile_grid(
    frame: np.ndarray,
    tiles: Sequence[Tuple[int, int, int, int]],
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    for x1, y1, x2, y2 in tiles:
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1, cv2.LINE_AA)

    return out


def draw_boxes(
    frame: np.ndarray,
    detections: Sequence[Detection],
    label: bool = False,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(1.0, w / 1920.0)

    font_scale = max(0.26, min(0.58, 0.22 + 0.10 * scale))
    thickness = max(1, int(round(scale * 0.8)))

    for det in detections:
        color = CLASS_COLORS_BGR.get(det.cls, (0, 255, 0))
        x1, y1, x2, y2 = det.bbox

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        if label:
            track_txt = f"#{det.track_id}" if det.track_id is not None else ""
            txt = f"{det.cls[0].upper()} {det.conf:.2f} {track_txt}".strip()

            ty = y1 - 4 if y1 > 14 else y2 + 12
            tx = x1

            approx_text_width = len(txt) * 8
            if tx + approx_text_width > w:
                tx = max(0, w - approx_text_width)

            cv2.putText(
                out,
                txt,
                (tx, int(ty)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )

    return out


def draw_stage_label(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    scale = max(1.0, w / 1920.0)

    bar_h = int(max(24, 34 * scale))
    cv2.rectangle(out, (0, 0), (w, bar_h), (0, 0, 0), -1)

    font_scale = max(0.35, min(0.90, 0.30 + 0.12 * scale))
    thickness = max(1, int(round(scale)))

    cv2.putText(
        out,
        label,
        (8, int(bar_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return out


def draw_annotations(
    frame: np.ndarray,
    detections: Sequence[Detection],
    gt_boxes: Sequence[Tuple[Tuple[int, int, int, int], str]],
    cfg: Config,
    stats: Dict[str, Any],
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(1.0, w / 1920.0)

    if cfg.use_tiling and stats.get("show_tile_grid", False):
        tiles = make_tiles(w, h, cfg.tile_size, cfg.tile_overlap)
        for x1, y1, x2, y2 in tiles:
            cv2.line(out, (x1, 0), (x1, h - 1), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(out, (0, y1), (w - 1, y1), (255, 255, 255), 1, cv2.LINE_AA)

    out = draw_boxes(out, detections, label=True)

    panel_h = min(int(138 * scale), h)
    panel_w = min(int(520 * scale), w)

    if panel_h > 10 and panel_w > 10:
        panel = out[0:panel_h, 0:panel_w].copy()
        cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (0, 0, 0), -1)
        out[0:panel_h, 0:panel_w] = cv2.addWeighted(
            out[0:panel_h, 0:panel_w],
            0.42,
            panel,
            0.58,
            0.0,
        )

        font_scale = max(0.32, min(0.82, 0.28 + 0.13 * scale))
        thickness = max(1, int(round(scale)))
        line_height = int(22 * scale)
        y = int(24 * scale)

        lines = [
            "ADVANCED UAV TINY-OBJECT PIPELINE",
            f"RES {w}x{h}  FPS {stats.get('fps', '?')}",
            f"GT {stats.get('gt', 0)}  PROP {stats.get('proposals', 0)}",
            f"MERGED {stats.get('merged', 0)}  TRACKED {stats.get('tracked', 0)}",
            f"THR {cfg.threshold:.2f}  SENS {cfg.sensitivity:.2f}",
            f"TILE {cfg.tile_size}px  OVERLAP {cfg.tile_overlap:.2f}",
            f"LATENCY {stats.get('latency_ms', 0.0):.1f} ms",
        ]

        for line in lines:
            cv2.putText(
                out,
                line,
                (int(10 * scale), y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            y += line_height

    return out


def save_stage_image(tmp_dir: Path, name: str, image: np.ndarray) -> str:
    path = tmp_dir / f"{name}.png"
    ok = cv2.imwrite(str(path), image)
    return str(path) if ok and path.exists() else ""


# =============================================================================
# Evaluation
# =============================================================================


def compute_ap_for_class(
    gt_cls: Sequence[Dict[str, Any]],
    det_cls: Sequence[Dict[str, Any]],
) -> Optional[float]:
    if not gt_cls:
        return None

    if not det_cls:
        return 0.0

    gt_by_image: Dict[int, List[int]] = {}
    for idx, gt in enumerate(gt_cls):
        gt_by_image.setdefault(int(gt["image_id"]), []).append(idx)

    matched = [False] * len(gt_cls)
    tp_list: List[int] = []
    fp_list: List[int] = []

    det_sorted = sorted(det_cls, key=lambda d: float(d["conf"]), reverse=True)

    for det in det_sorted:
        img_id = int(det["image_id"])
        chosen = -1
        best_iou = 0.0

        for idx in gt_by_image.get(img_id, []):
            if matched[idx]:
                continue

            score = iou(det["bbox"], gt_cls[idx]["bbox"])
            if score >= 0.5 and score > best_iou:
                best_iou = score
                chosen = idx

        if chosen >= 0:
            matched[chosen] = True
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)

    recall = tp_cum / float(len(gt_cls))
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    interp_recall = np.linspace(0.0, 1.0, 101)
    ap_values = np.interp(interp_recall, recall, precision)

    return float(np.mean(ap_values))


def evaluate_detection_results(
    gt_per_image: Sequence[Sequence[Tuple[Tuple[int, int, int, int], str]]],
    det_per_image: Sequence[Sequence[Detection]],
    num_images: int,
) -> Dict[str, Any]:
    gt_records: List[Dict[str, Any]] = []
    det_records: List[Dict[str, Any]] = []

    for img_id, gts in enumerate(gt_per_image):
        for bbox, cls in gts:
            area = bbox_area(bbox)
            gt_records.append(
                {
                    "image_id": int(img_id),
                    "bbox": bbox,
                    "cls": cls,
                    "area": int(area),
                    "tiny": bool(area < TINY_AREA),
                    "small": bool(area < SMALL_AREA),
                    "medium": bool(SMALL_AREA <= area < MEDIUM_AREA),
                }
            )

    for img_id, dets in enumerate(det_per_image):
        for det in dets:
            det_records.append(
                {
                    "image_id": int(img_id),
                    "bbox": det.bbox,
                    "cls": det.cls,
                    "conf": float(det.conf),
                }
            )

    all_classes = sorted({str(g["cls"]) for g in gt_records}) if gt_records else list(CLASS_NAMES)
    small_gt = [g for g in gt_records if bool(g["small"])]
    total_small = len(small_gt)

    base_metrics: Dict[str, Any] = {
        "frames": float(max(0, num_images)),
        "gt_objects_total": float(len(gt_records)),
        "small_objects_total": float(total_small),
        "avg_gt_per_frame": float(len(gt_records) / float(max(1, num_images))),
        "avg_small_gt_per_frame": float(total_small / float(max(1, num_images))),
        "avg_detections_per_frame": float(len(det_records) / float(max(1, num_images))),
        "mAP_small": 0.0,
        "recall_at_fppi_1": 0.0,
        "precision_at_fppi_1": 0.0,
        "ap_tiny": 0.0,
        "ap_small_only": 0.0,
        "ap_medium": 0.0,
    }

    for cls in CLASS_NAMES:
        base_metrics[f"ap_{cls}"] = 0.0

    if num_images <= 0 or total_small == 0:
        return base_metrics

    small_classes = sorted({str(g["cls"]) for g in small_gt})
    class_aps: Dict[str, float] = {}

    for cls in small_classes:
        gt_cls = [g for g in small_gt if str(g["cls"]) == cls]
        det_cls = [d for d in det_records if str(d["cls"]) == cls]
        ap = compute_ap_for_class(gt_cls, det_cls)
        class_aps[cls] = float(ap) if ap is not None else 0.0

    base_metrics["mAP_small"] = float(np.mean(list(class_aps.values()))) if class_aps else 0.0

    for cls in CLASS_NAMES:
        base_metrics[f"ap_{cls}"] = float(class_aps.get(cls, 0.0))

    def ap_for_bucket(predicate) -> float:
        aps: List[float] = []

        for cls in all_classes:
            gt_cls = [g for g in gt_records if str(g["cls"]) == cls and bool(predicate(g))]
            if not gt_cls:
                continue

            det_cls = [d for d in det_records if str(d["cls"]) == cls]
            ap = compute_ap_for_class(gt_cls, det_cls)

            if ap is not None:
                aps.append(float(ap))

        return float(np.mean(aps)) if aps else 0.0

    base_metrics["ap_tiny"] = ap_for_bucket(lambda g: bool(g["tiny"]))
    base_metrics["ap_small_only"] = ap_for_bucket(
        lambda g: bool(not g["tiny"] and g["small"])
    )
    base_metrics["ap_medium"] = ap_for_bucket(lambda g: bool(g["medium"]))

    det_sorted = sorted(det_records, key=lambda d: float(d["conf"]), reverse=True)

    gt_lookup: Dict[Tuple[int, str], List[int]] = {}
    for idx, gt in enumerate(small_gt):
        key = (int(gt["image_id"]), str(gt["cls"]))
        gt_lookup.setdefault(key, []).append(idx)

    matched = [False] * len(small_gt)
    tp = 0
    fp = 0

    best_recall = 0.0
    best_precision = 0.0
    allowed_fp = float(num_images)

    for det in det_sorted:
        key = (int(det["image_id"]), str(det["cls"]))
        candidates = gt_lookup.get(key, [])

        chosen = -1
        best_iou = 0.0

        for idx in candidates:
            if matched[idx]:
                continue

            score = iou(det["bbox"], small_gt[idx]["bbox"])
            if score >= 0.5 and score > best_iou:
                best_iou = score
                chosen = idx

        if chosen >= 0:
            matched[chosen] = True
            tp += 1
        else:
            fp += 1

        if fp <= allowed_fp:
            recall = tp / float(total_small)
            precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0

            if recall > best_recall:
                best_recall = recall
                best_precision = precision
        else:
            break

    base_metrics["recall_at_fppi_1"] = float(best_recall)
    base_metrics["precision_at_fppi_1"] = float(best_precision)

    return base_metrics


# =============================================================================
# Cached demo generator
# =============================================================================


@st.cache_data(show_spinner="Building advanced real-image UAV pipeline...")
def generate_advanced_demo(
    cfg_items: Tuple[Tuple[str, Any], ...],
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], Dict[str, int]]:
    cfg = Config(**dict(cfg_items))

    num_frames = max(1, int(round(cfg.fps * cfg.duration_s)))
    sample_frame = max(0, num_frames // 2)

    tmp_dir = Path(tempfile.mkdtemp(prefix="uav_advanced_real_"))
    video_path = tmp_dir / "tiny_object_real_images.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        float(cfg.fps),
        (cfg.width, cfg.height),
    )

    if not writer.isOpened():
        video_path = tmp_dir / "tiny_object_real_images.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            float(cfg.fps),
            (cfg.width, cfg.height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                "OpenCV VideoWriter could not open an MP4 or AVI output file. "
                "Install a full OpenCV build with video encoding support."
            )

    events: List[Dict[str, Any]] = []

    sprites: Dict[str, Optional[np.ndarray]] = {}
    sprite_status = {"person": False, "animal": False, "vehicle": False}

    if cfg.use_real_sprites:
        for cls, key in [
            ("person", cfg.person_sprite_key),
            ("animal", cfg.animal_sprite_key),
            ("vehicle", cfg.vehicle_sprite_key),
        ]:
            img = resolve_sprite(key)
            sprites[cls] = img
            sprite_status[cls] = img is not None

        if not any(sprite_status.values()):
            events.append(
                {
                    "frame": 0,
                    "type": "sprite_fallback",
                    "severity": "info",
                    "message": "Real images unavailable; using synthetic shapes fallback.",
                }
            )
    else:
        sprites = {"person": None, "animal": None, "vehicle": None}

    scene = SyntheticAerialScene(cfg, sprites)
    tracker = TinyTracker(
        iou_threshold=cfg.tracker_iou,
        max_misses=cfg.max_tracker_misses,
        alpha=cfg.tracker_alpha,
        min_hits=cfg.min_track_hits,
    )

    det_rng = np.random.default_rng(cfg.seed + 17)

    per_frame: List[Dict[str, Any]] = []
    gt_frames: List[List[Tuple[Tuple[int, int, int, int], str]]] = []
    det_frames: List[List[Detection]] = []
    latency_list: List[float] = []

    stage_paths: Dict[str, str] = {}

    budget_ms = 1000.0 / float(max(1, cfg.fps))

    for frame_idx in range(num_frames):
        t0 = time.perf_counter()

        frame, gt_boxes, empty_flag = scene.step(cfg)

        if empty_flag:
            events.append(
                {
                    "frame": int(frame_idx),
                    "type": "empty_frame",
                    "severity": "warning",
                    "message": "Synthetic empty/drop frame injected; tracker enters recovery mode",
                }
            )

        if cfg.use_tiling:
            tiles = make_tiles(cfg.width, cfg.height, cfg.tile_size, cfg.tile_overlap)
        else:
            tiles = [(0, 0, cfg.width, cfg.height)]

        proposals: List[Detection] = []

        for tile in tiles:
            x1, y1, x2, y2 = tile
            tile_img = frame[y1:y2, x1:x2]

            if tile_img.size == 0:
                continue

            tile_dets = simulate_tile_detections(
                tile_img=tile_img,
                tile_box=tile,
                gt_boxes=gt_boxes,
                rng=det_rng,
                cfg=cfg,
            )

            proposals.extend(tile_dets)

        if cfg.use_wbf:
            merged = weighted_box_fusion(proposals, iou_threshold=cfg.nms_iou)
        else:
            merged = non_max_suppression(proposals, iou_threshold=cfg.nms_iou)

        merged = apply_threshold_and_calibration(merged, cfg)
        merged = suppress_osd_false_alarms(
            detections=merged,
            cfg=cfg,
            width=cfg.width,
            height=cfg.height,
            events=events,
            frame_idx=frame_idx,
        )

        if cfg.use_tracker:
            tracked = tracker.update(merged, frame_idx)
        else:
            tracked = non_max_suppression(merged, iou_threshold=cfg.nms_iou)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        latency_list.append(float(latency_ms))

        if latency_ms > budget_ms * 1.75:
            events.append(
                {
                    "frame": int(frame_idx),
                    "type": "latency_spike",
                    "severity": "warning",
                    "message": f"Frame latency {latency_ms:.1f} ms exceeded 1.75x budget",
                    "latency_ms": round(float(latency_ms), 2),
                    "budget_ms": round(float(budget_ms), 2),
                }
            )

        stats = {
            "fps": cfg.fps,
            "gt": len(gt_boxes),
            "proposals": len(proposals),
            "merged": len(merged),
            "tracked": len(tracked),
            "latency_ms": float(latency_ms),
            "show_tile_grid": False,
        }

        final_frame = draw_annotations(
            frame=frame,
            detections=tracked,
            gt_boxes=gt_boxes,
            cfg=cfg,
            stats=stats,
        )

        writer.write(final_frame)

        gt_frames.append(gt_boxes)
        det_frames.append(tracked)

        per_frame.append(
            {
                "frame": int(frame_idx),
                "gt": int(len(gt_boxes)),
                "proposals": int(len(proposals)),
                "merged": int(len(merged)),
                "tracked": int(len(tracked)),
                "latency_ms": round(float(latency_ms), 2),
                "empty": bool(empty_flag),
            }
        )

        if frame_idx == sample_frame:
            stage_raw = draw_stage_label(frame.copy(), "Stage 1: Raw synthetic aerial scene")
            stage_tiles = draw_stage_label(draw_tile_grid(frame, tiles), "Stage 2: SAHI-style tiling")
            stage_proposals = draw_stage_label(draw_boxes(frame, proposals, label=False), "Stage 3: Tiny-object proposals")
            stage_merged = draw_stage_label(draw_boxes(frame, merged, label=True), "Stage 4: WBF/NMS + threshold + OSD suppression")
            stage_tracked = draw_stage_label(draw_boxes(frame, tracked, label=True), "Stage 5: Tracker-smoothed output")
            stage_final = draw_stage_label(final_frame, "Stage 6: Final annotated UAV feed")

            stage_paths = {
                "01_raw_scene": save_stage_image(tmp_dir, "01_raw_scene", stage_raw),
                "02_tiling": save_stage_image(tmp_dir, "02_tiling", stage_tiles),
                "03_proposals": save_stage_image(tmp_dir, "03_proposals", stage_proposals),
                "04_merged": save_stage_image(tmp_dir, "04_merged", stage_merged),
                "05_tracked": save_stage_image(tmp_dir, "05_tracked", stage_tracked),
                "06_final": save_stage_image(tmp_dir, "06_final", stage_final),
            }

    writer.release()

    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError("Video generation completed but output file is empty.")

    metrics = evaluate_detection_results(
        gt_per_image=gt_frames,
        det_per_image=det_frames,
        num_images=num_frames,
    )

    if latency_list:
        metrics["latency_avg_ms"] = float(np.mean(latency_list))
        metrics["latency_p50_ms"] = float(np.percentile(latency_list, 50))
        metrics["latency_p95_ms"] = float(np.percentile(latency_list, 95))
        metrics["estimated_fps_from_p95"] = float(1000.0 / max(1.0, metrics["latency_p95_ms"]))
    else:
        metrics["latency_avg_ms"] = 0.0
        metrics["latency_p50_ms"] = 0.0
        metrics["latency_p95_ms"] = 0.0
        metrics["estimated_fps_from_p95"] = 0.0

    metrics["latency_budget_ms"] = float(budget_ms)
    metrics["target_15fps_budget_ms"] = float(TARGET_LATENCY_MS)
    metrics["empty_frames"] = int(sum(1 for p in per_frame if bool(p["empty"])))
    metrics["osd_suppressions"] = int(sum(1 for e in events if e.get("type") == "osd_suppression"))
    metrics["latency_spikes"] = int(sum(1 for e in events if e.get("type") == "latency_spike"))

    metrics["duration_s"] = float(cfg.duration_s)
    metrics["requested_fps"] = float(cfg.fps)
    metrics["resolution"] = f"{cfg.width}x{cfg.height}"
    metrics["tiling_enabled"] = bool(cfg.use_tiling)
    metrics["wbf_enabled"] = bool(cfg.use_wbf)
    metrics["tracker_enabled"] = bool(cfg.use_tracker)

    metrics["real_person_image"] = bool(sprite_status["person"])
    metrics["real_animal_image"] = bool(sprite_status["animal"])
    metrics["real_vehicle_image"] = bool(sprite_status["vehicle"])

    confidence_hist = {cls: 0 for cls in CLASS_NAMES}

    for dets in det_frames:
        for det in dets:
            if det.cls in confidence_hist:
                confidence_hist[det.cls] += 1

    return (
        str(video_path),
        metrics,
        events,
        stage_paths,
        per_frame,
        confidence_hist,
    )


# =============================================================================
# Live playback
# =============================================================================


def play_live_stream(video_path: str, fps: int, expected_frames: int) -> None:
    placeholder = st.empty()
    progress = st.progress(0.0, text="Starting real-image synthetic UAV feed...")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        progress.empty()
        st.warning(
            "Could not open the generated video for live playback. "
            "The saved video file is still available below."
        )
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frame_count = max(1, expected_frames)

    idx = 0
    start = time.perf_counter()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        idx += 1

        placeholder.image(
            frame,
            channels="BGR",
            use_container_width=True,
        )

        progress.progress(
            min(1.0, idx / float(frame_count)),
            text=f"Live real-image UAV feed: frame {idx}/{frame_count}",
        )

        target_time = idx / float(max(1, fps))
        sleep_for = target_time - (time.perf_counter() - start)

        if sleep_for > 0:
            time.sleep(sleep_for)

    cap.release()
    progress.empty()


# =============================================================================
# Export helpers
# =============================================================================


def events_to_ndjson(events: List[Dict[str, Any]]) -> bytes:
    if not events:
        return b""
    return ("\n".join(json.dumps(e, default=str) for e in events)).encode("utf-8")


def metrics_to_json(metrics: Dict[str, Any]) -> bytes:
    return json.dumps(metrics, indent=2, default=str).encode("utf-8")


def per_frame_to_csv(per_frame: List[Dict[str, Any]]) -> bytes:
    if not per_frame:
        return b""

    header = "frame,gt,proposals,merged,tracked,latency_ms,empty"
    rows = [header]

    for row in per_frame:
        rows.append(
            ",".join(
                [
                    str(row.get("frame", "")),
                    str(row.get("gt", "")),
                    str(row.get("proposals", "")),
                    str(row.get("merged", "")),
                    str(row.get("tracked", "")),
                    str(row.get("latency_ms", "")),
                    "true" if row.get("empty") else "false",
                ]
            )
        )

    return ("\n".join(rows)).encode("utf-8")


# =============================================================================
# Streamlit UI
# =============================================================================


def main() -> None:
    st.set_page_config(
        page_title="Advanced Real-Image Tiny Object Drone Demo",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🛰️ Advanced Tiny Vehicle / Person / Animal Detector — Real-Image UAV Demo")
    st.caption(
        "This app generates an aerial drone scene, loads real person / animal / vehicle "
        "images automatically when available, simulates a tiny-object detection pipeline "
        "step by step, plays a live feed immediately, and produces a downloadable video "
        "file. No API key is required."
    )

    with st.sidebar:
        st.header("1. Scene")

        resolution = st.selectbox(
            "Render resolution",
            options=["1280x720", "1920x1080", "2560x1440", "3840x2160"],
            index=1,
            help=(
                "The demo is designed around high-altitude UAV inspection. "
                "1080p is default for smoother browser playback."
            ),
        )

        width, height = map(int, resolution.split("x"))
        fps = st.slider("Playback FPS", min_value=10, max_value=30, value=25)
        duration_s = st.slider("Clip duration seconds", min_value=3, max_value=20, value=8)
        object_count = st.slider("Objects per scene", min_value=20, max_value=220, value=90)
        seed = int(st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1))

        st.header("2. Detector")

        sensitivity = st.slider(
            "Detector sensitivity",
            min_value=0.60,
            max_value=1.30,
            value=1.00,
            step=0.05,
        )

        threshold = st.slider(
            "Confidence threshold",
            min_value=0.15,
            max_value=0.75,
            value=0.30,
            step=0.05,
        )

        use_tiling = st.checkbox("Enable SAHI-style tiling", value=True)

        tile_size = st.select_slider(
            "Tile size px",
            options=[320, 480, 640, 960, 1280, 1920],
            value=640,
            disabled=not use_tiling,
        )

        tile_overlap = st.slider(
            "Tile overlap ratio",
            min_value=0.0,
            max_value=0.45,
            value=0.20,
            step=0.05,
            disabled=not use_tiling,
        )

        use_wbf = st.checkbox("Use WBF merging", value=True)
        calibrate = st.checkbox("Use confidence calibration", value=True)

        nms_iou = st.slider(
            "NMS / WBF IoU",
            min_value=0.30,
            max_value=0.70,
            value=0.45,
            step=0.05,
        )

        st.header("3. Temporal smoothing")

        use_tracker = st.checkbox("Enable tracker", value=True)

        min_track_hits = st.slider(
            "Min track hits",
            min_value=1,
            max_value=5,
            value=2,
            disabled=not use_tracker,
        )

        tracker_iou = st.slider(
            "Tracker IoU",
            min_value=0.20,
            max_value=0.50,
            value=0.30,
            step=0.05,
            disabled=not use_tracker,
        )

        st.header("4. Stress / edge cases")

        motion_blur = st.checkbox("Motion blur / rolling shutter", value=True)
        compression_noise = st.checkbox("Compression / sensor noise", value=True)
        glare = st.checkbox("Sun glare", value=False)
        shadows = st.checkbox("Object shadows", value=True)
        gps_overlay = st.checkbox("GPS / OSD overlay text", value=True)
        empty_frames = st.checkbox("Inject empty / drop frames", value=False)

        st.header("5. Real images")

        use_real_sprites = st.checkbox(
            "Use real person / animal / vehicle images",
            value=True,
            help=(
                "The app tries public real-image URLs first, then Wikimedia Commons, "
                "then Wikipedia. You can also upload your own images."
            ),
        )

        uploaded_person = st.file_uploader(
            "Optional: upload real person image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="upload_person",
            disabled=not use_real_sprites,
        )

        uploaded_animal = st.file_uploader(
            "Optional: upload real animal image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="upload_animal",
            disabled=not use_real_sprites,
        )

        uploaded_vehicle = st.file_uploader(
            "Optional: upload real vehicle image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="upload_vehicle",
            disabled=not use_real_sprites,
        )

        person_path = _save_upload_to_temp(uploaded_person, "person") if use_real_sprites else None
        animal_path = _save_upload_to_temp(uploaded_animal, "animal") if use_real_sprites else None
        vehicle_path = _save_upload_to_temp(uploaded_vehicle, "vehicle") if use_real_sprites else None

        person_key = f"file:{person_path}" if person_path else DEFAULT_PERSON_SPRITE_KEY
        animal_key = f"file:{animal_path}" if animal_path else DEFAULT_ANIMAL_SPRITE_KEY
        vehicle_key = f"file:{vehicle_path}" if vehicle_path else DEFAULT_VEHICLE_SPRITE_KEY

        if use_real_sprites:
            with st.expander("Preview real sprites"):
                preview_cols = st.columns(3)

                for col, label, key in [
                    (preview_cols[0], "Person", person_key),
                    (preview_cols[1], "Animal", animal_key),
                    (preview_cols[2], "Vehicle", vehicle_key),
                ]:
                    with col:
                        img = resolve_sprite(key)
                        disp = _display_image(img)
                        if disp is not None:
                            st.image(disp, caption=label, width=80)
                        else:
                            st.caption(f"{label}: fallback shape")

        st.divider()
        replay = st.button("Replay live stream", type="secondary")

        st.caption(
            "Run locally:\n\n"
            "`pip install -r requirements.txt`\n\n"
            "`streamlit run app.py`"
        )

    if replay:
        st.session_state["played_for"] = None

    cfg_dict = {
        "width": int(width),
        "height": int(height),
        "fps": int(fps),
        "duration_s": int(duration_s),
        "object_count": int(object_count),
        "seed": int(seed),
        "sensitivity": float(sensitivity),
        "threshold": float(threshold),
        "use_tiling": bool(use_tiling),
        "tile_size": int(tile_size),
        "tile_overlap": float(tile_overlap),
        "use_wbf": bool(use_wbf),
        "nms_iou": float(nms_iou),
        "use_tracker": bool(use_tracker),
        "tracker_iou": float(tracker_iou),
        "min_track_hits": int(min_track_hits),
        "calibrate": bool(calibrate),
        "motion_blur": bool(motion_blur),
        "compression_noise": bool(compression_noise),
        "glare": bool(glare),
        "shadows": bool(shadows),
        "gps_overlay": bool(gps_overlay),
        "empty_frames": bool(empty_frames),
        "use_real_sprites": bool(use_real_sprites),
        "person_sprite_key": str(person_key),
        "animal_sprite_key": str(animal_key),
        "vehicle_sprite_key": str(vehicle_key),
    }

    params_key = json.dumps(cfg_dict, sort_keys=True)
    cfg_items = tuple(sorted(cfg_dict.items()))

    num_frames = max(1, int(round(fps * duration_s)))

    try:
        video_path, metrics, events, stage_paths, per_frame, confidence_hist = (
            generate_advanced_demo(cfg_items)
        )
    except Exception as exc:
        st.error(f"Demo generation failed: {exc}")
        st.stop()

    if not Path(video_path).exists():
        st.cache_data.clear()

        try:
            video_path, metrics, events, stage_paths, per_frame, confidence_hist = (
                generate_advanced_demo(cfg_items)
            )
        except Exception as exc:
            st.error(f"Demo regeneration failed: {exc}")
            st.stop()

    tab_live, tab_pipeline, tab_metrics, tab_edges, tab_exports, tab_guide = st.tabs(
        [
            "🎥 Live Demo",
            "🧠 Pipeline Steps",
            "📊 Metrics",
            "🧪 Edge Cases",
            "📦 Exports",
            "📘 Guide",
        ]
    )

    with tab_live:
        st.subheader("Live synthetic UAV feed with real-image sprites")

        real_cols = st.columns(3)
        real_cols[0].metric("Person image", "Loaded" if metrics.get("real_person_image") else "Fallback")
        real_cols[1].metric("Animal image", "Loaded" if metrics.get("real_animal_image") else "Fallback")
        real_cols[2].metric("Vehicle image", "Loaded" if metrics.get("real_vehicle_image") else "Fallback")

        if st.session_state.get("played_for") != params_key:
            play_live_stream(video_path, fps=fps, expected_frames=num_frames)
            st.session_state["played_for"] = params_key
        else:
            st.caption(
                "Live playback completed for the current settings. "
                "Use **Replay live stream** in the sidebar to run it again."
            )

        video_file = Path(video_path)
        file_size = video_file.stat().st_size if video_file.exists() else 0

        if 0 < file_size < 80_000_000:
            st.video(
                video_file.read_bytes(),
                autoplay=True,
                muted=True,
                loop=False,
            )
        else:
            st.video(
                str(video_file),
                autoplay=True,
                muted=True,
                loop=False,
            )

        mime = "video/mp4" if video_file.suffix.lower() == ".mp4" else "video/x-msvideo"

        with video_file.open("rb") as fh:
            st.download_button(
                "Download generated video",
                data=fh,
                file_name=video_file.name,
                mime=mime,
            )

        st.info(
            "This demo generates the video locally, then plays it as a live-style feed. "
            "It tries to use real person / animal / vehicle images. If online images are "
            "blocked, upload your own images in the sidebar or the app falls back to "
            "synthetic shapes."
        )

    with tab_pipeline:
        st.subheader("Step-by-step advanced pipeline")

        st.markdown(
            """
            This app simulates a production-style small-object UAV detection workflow:

            1. **Ingestion / synthetic scene generation**  
               Creates a high-altitude aerial scene with roads, fields, river, buildings,
               tiny vehicles, people, and animals.

            2. **Real-image sprite loading**  
               Loads real person / animal / vehicle images from public sources or uploads,
               then composites them into the moving scene.

            3. **Pre-processing / SAHI-style tiling**  
               Splits the frame into overlapping tiles so tiny objects receive more
               effective resolution.

            4. **Tiny-object proposal simulation**  
               Generates class-wise detections with size-dependent confidence,
               localization jitter, tile-edge penalties, and false positives.

            5. **Overlap-aware merging**  
               Uses Weighted Box Fusion or NMS to merge duplicate tile detections.

            6. **Thresholding, calibration, and OSD suppression**  
               Applies class-specific thresholds and suppresses likely GPS/telemetry
               overlay false alarms.

            7. **Tracker-assisted temporal smoothing**  
               Uses a lightweight IoU tracker to stabilize boxes across frames.

            8. **Annotation, metrics, and export**  
               Renders HUD overlays, writes an MP4/AVI file, and records events.
            """
        )

        st.divider()
        st.subheader("Intermediate stage previews")

        stage_items = [
            ("01_raw_scene", "Stage 1: Raw synthetic aerial scene"),
            ("02_tiling", "Stage 2: SAHI-style tiling"),
            ("03_proposals", "Stage 3: Tiny-object proposals"),
            ("04_merged", "Stage 4: Merged / filtered detections"),
            ("05_tracked", "Stage 5: Tracker-smoothed detections"),
            ("06_final", "Stage 6: Final annotated feed"),
        ]

        cols = st.columns(2)

        for idx, (key, caption) in enumerate(stage_items):
            path = stage_paths.get(key, "")

            with cols[idx % 2]:
                if path and Path(path).exists():
                    st.image(path, caption=caption, use_container_width=True)
                else:
                    st.warning(f"Stage image unavailable: {caption}")

        st.divider()
        st.subheader("Per-frame pipeline statistics")
        st.dataframe(per_frame, use_container_width=True)

    with tab_metrics:
        st.subheader("Simulated acceptance metrics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "mAP-small",
            f"{metrics.get('mAP_small', 0.0):.3f}",
            delta=f"{metrics.get('mAP_small', 0.0) - 0.45:+.3f}",
        )

        col2.metric(
            "Recall @ 1 FPPI",
            f"{metrics.get('recall_at_fppi_1', 0.0):.3f}",
            delta=f"{metrics.get('recall_at_fppi_1', 0.0) - 0.80:+.3f}",
        )

        col3.metric(
            "Precision @ 1 FPPI",
            f"{metrics.get('precision_at_fppi_1', 0.0):.3f}",
        )

        col4.metric(
            "Detections / frame",
            f"{metrics.get('avg_detections_per_frame', 0.0):.1f}",
        )

        failing_gates: List[str] = []

        if metrics.get("mAP_small", 0.0) < 0.45:
            failing_gates.append("mAP-small < 0.45")

        if metrics.get("recall_at_fppi_1", 0.0) < 0.80:
            failing_gates.append("Recall @ 1 FPPI < 0.80")

        if not failing_gates:
            st.success(
                "Simulated release gates pass: mAP-small ≥ 0.45 and recall ≥ 0.80 at 1 FPPI."
            )
        else:
            st.warning(
                "Simulated release gates need tuning: " + "; ".join(failing_gates)
            )

        st.divider()
        st.subheader("Per-class small-object AP")

        class_cols = st.columns(3)

        for idx, cls in enumerate(CLASS_NAMES):
            class_cols[idx].metric(
                cls.capitalize(),
                f"{metrics.get(f'ap_{cls}', 0.0):.3f}",
            )

        st.divider()
        st.subheader("Per-size breakdown")

        size_cols = st.columns(3)

        size_cols[0].metric("AP tiny < 16x16", f"{metrics.get('ap_tiny', 0.0):.3f}")
        size_cols[1].metric("AP small 16x16-32x32", f"{metrics.get('ap_small_only', 0.0):.3f}")
        size_cols[2].metric("AP medium 32x32-96x96", f"{metrics.get('ap_medium', 0.0):.3f}")

        st.divider()
        st.subheader("Latency diagnostics")

        lat_col1, lat_col2, lat_col3, lat_col4 = st.columns(4)

        lat_col1.metric("Avg latency", f"{metrics.get('latency_avg_ms', 0.0):.1f} ms")
        lat_col2.metric("P50 latency", f"{metrics.get('latency_p50_ms', 0.0):.1f} ms")
        lat_col3.metric("P95 latency", f"{metrics.get('latency_p95_ms', 0.0):.1f} ms")
        lat_col4.metric(
            "Estimated FPS from P95",
            f"{metrics.get('estimated_fps_from_p95', 0.0):.1f}",
        )

        st.caption(
            "Latency here is measured inside the Python/Streamlit demo process. "
            "It is not a TensorRT/RTX production benchmark. The original hardware target "
            "is 15 FPS on an RTX-class GPU, which corresponds to a 66.7 ms/frame budget."
        )

        st.divider()
        st.subheader("Confidence histogram by class")

        st.bar_chart(confidence_hist)

        st.caption(
            "These metrics are computed from synthetic ground truth and simulated detector "
            "outputs. They validate the demo pipeline, not a trained production model."
        )

    with tab_edges:
        st.subheader("Edge-case and robustness report")

        osd_suppressions = int(metrics.get("osd_suppressions", 0))
        empty_frames_count = int(metrics.get("empty_frames", 0))
        latency_spikes = int(metrics.get("latency_spikes", 0))

        edge_rows = [
            {
                "Scenario": "Motion blur",
                "Enabled": bool(motion_blur),
                "Expected behavior": "No crash; graceful degraded detections; tracker helps stabilize.",
                "Observed": "Simulated in scene renderer.",
                "Status": "Configured",
            },
            {
                "Scenario": "Shadows",
                "Enabled": bool(shadows),
                "Expected behavior": "Shadows should not become high-confidence objects.",
                "Observed": "Shadow pixels are drawn but not added to ground truth.",
                "Status": "Configured",
            },
            {
                "Scenario": "Partially cropped tiles",
                "Enabled": bool(use_tiling),
                "Expected behavior": "Tile-edge penalty and WBF reduce boundary fragmentation.",
                "Observed": "Tile-edge confidence penalty active." if use_tiling else "Tiling disabled.",
                "Status": "Configured" if use_tiling else "Disabled",
            },
            {
                "Scenario": "Empty frames",
                "Enabled": bool(empty_frames),
                "Expected behavior": "System recovers without crash; tracker prunes stale tracks.",
                "Observed": f"{empty_frames_count} empty frame(s) injected.",
                "Status": "Recovered" if empty_frames else "Not tested",
            },
            {
                "Scenario": "GPS / OSD overlay text",
                "Enabled": bool(gps_overlay),
                "Expected behavior": "Low-confidence tiny detections in OSD bands are suppressed.",
                "Observed": f"{osd_suppressions} OSD false alarm suppression event(s).",
                "Status": "Suppressed" if gps_overlay else "Not tested",
            },
            {
                "Scenario": "Compression / sensor noise",
                "Enabled": bool(compression_noise),
                "Expected behavior": "More false positives, lower confidence, no crash.",
                "Observed": "Noise and resampling artifacts injected.",
                "Status": "Configured",
            },
            {
                "Scenario": "Sun glare",
                "Enabled": bool(glare),
                "Expected behavior": "Reduced contrast and more false positives in bright region.",
                "Observed": "Glare overlay injected." if glare else "Disabled.",
                "Status": "Configured" if glare else "Disabled",
            },
            {
                "Scenario": "Real image loading",
                "Enabled": bool(use_real_sprites),
                "Expected behavior": "Use public real images or uploaded images; fallback if unavailable.",
                "Observed": (
                    f"person={metrics.get('real_person_image', False)}, "
                    f"animal={metrics.get('real_animal_image', False)}, "
                    f"vehicle={metrics.get('real_vehicle_image', False)}"
                ),
                "Status": "Active" if use_real_sprites else "Disabled",
            },
            {
                "Scenario": "Latency spikes",
                "Enabled": True,
                "Expected behavior": "Log warning when frame processing exceeds budget.",
                "Observed": f"{latency_spikes} latency spike event(s).",
                "Status": "Monitored",
            },
        ]

        st.dataframe(edge_rows, use_container_width=True)

        st.divider()
        st.subheader("Recent events")

        if events:
            st.json(events[-80:])
        else:
            st.success("No warning or suppression events were recorded for this configuration.")

    with tab_exports:
        st.subheader("Export artifacts")

        video_file = Path(video_path)

        exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)

        with exp_col1:
            with video_file.open("rb") as fh:
                st.download_button(
                    "Download video",
                    data=fh,
                    file_name=video_file.name,
                    mime="video/mp4" if video_file.suffix.lower() == ".mp4" else "video/x-msvideo",
                )

        with exp_col2:
            st.download_button(
                "Download metrics JSON",
                data=metrics_to_json(metrics),
                file_name="metrics.json",
                mime="application/json",
            )

        with exp_col3:
            st.download_button(
                "Download events NDJSON",
                data=events_to_ndjson(events),
                file_name="pipeline_events.ndjson",
                mime="application/x-ndjson",
            )

        with exp_col4:
            st.download_button(
                "Download per-frame CSV",
                data=per_frame_to_csv(per_frame),
                file_name="per_frame_stats.csv",
                mime="text/csv",
            )

        st.divider()
        st.subheader("Metrics preview")
        st.json(metrics)

    with tab_guide:
        st.subheader("How to run")

        st.code(
            """
pip install -r requirements.txt
streamlit run app.py
            """.strip(),
            language="bash",
        )

        st.divider()
        st.subheader("Real images")

        st.markdown(
            """
            This version tries to use real images for people, animals, and vehicles.

            Sources used automatically:

            1. Direct real-image URLs  
            2. Wikimedia Commons search  
            3. Wikipedia article thumbnails  

            You can also upload your own JPG/PNG/WebP images in the sidebar for:

            - person
            - animal
            - vehicle

            If no online image is available, the app safely falls back to synthetic
            colored shapes so the demo still runs.
            """
        )

        st.divider()
        st.subheader("Important limitation")

        st.warning(
            "This app does not contain a trained neural network. The detector is simulated "
            "so the project can run immediately with no dataset, no weights, and no API key. "
            "The metrics validate the pipeline logic, not real-world model accuracy."
        )


if __name__ == "__main__":
    main()