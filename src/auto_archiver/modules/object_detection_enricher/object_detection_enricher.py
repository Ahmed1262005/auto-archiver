"""Object and logo detection on images and video frames, on CPU, with Hugging Face transformers detectors.

Objects come from a closed-set COCO detector (RT-DETRv2 by default); logos from an open-vocabulary detector
prompted with "logo" (OWLv2 by default). A logo box gets a brand when the OCR text inside it, or its SigLIP 2
embedding against a folder of reference logos, matches a known brand. Frames are sampled exactly like
ocr_enricher (source timestamps), so the OCR boxes line up with these boxes in a viewer.
"""

import difflib
import importlib.metadata
import os
import re
import threading
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from auto_archiver.core import Enricher, Metadata
from auto_archiver.modules.ocr_enricher import ocr_enricher as _ocr
from auto_archiver.utils.custom_logger import logger
from auto_archiver.utils.enricher_errors import EnricherInputError, EnricherServiceError

TOOL = "transformers"
LAYERS = ("objects", "logos")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

DEFAULTS = {
    "layers": LAYERS,
    "fps": 1.0,
    "max_frames": 0,
    "object_model": "PekingU/rtdetr_v2_r18vd",
    "object_threshold": 0.5,
    "logo_model": "google/owlv2-base-patch16-ensemble",
    "logo_prompts": ("logo",),
    "logo_threshold": 0.2,
    "max_logos_per_frame": 10,
    "brands": (),
    "brand_refs_dir": os.environ.get("OD_BRAND_REFS_DIR") or None,
    "brand_match": os.environ.get("OD_BRAND_MATCH", "1") != "0",
    "embed_model": "google/siglip2-base-patch16-224",
    "brand_match_threshold": 0.8,
    "logo_ocr": True,
    "wordmarks": True,
    "cpu_threads": 0,
    "model_dir": os.environ.get("OD_MODEL_DIR") or None,
    "timeout": 600,
}

_MODELS = {}
_MODELS_LOCK = threading.Lock()
_REFS = {}


def analyze(path: str, **opts) -> dict:
    """Detect objects and logos in an image or a video file. Options (all optional): layers, fps, max_frames,
    object_model, object_threshold, logo_model, logo_prompts, logo_threshold, max_logos_per_frame, brands,
    brand_refs_dir, brand_match, embed_model, brand_match_threshold, logo_ocr, wordmarks, cpu_threads, model_dir, timeout,
    stills=[(timestamp, image_path), ...] to analyse given stills instead of sampling the video.
    Boxes are [x0, y0, x1, y1] normalised to 0..1 of the frame. Raises EnricherInputError / EnricherServiceError."""
    unknown = set(opts) - set(DEFAULTS) - {"stills"}
    if unknown:
        raise TypeError(f"analyze() got unknown options: {sorted(unknown)}")
    o = {**DEFAULTS, **{k: v for k, v in opts.items() if v is not None}}
    layers = _layers(o["layers"])
    fps = float(o["fps"])
    if fps <= 0:
        raise ValueError("fps must be > 0")
    prompts = [o["logo_prompts"]] if isinstance(o["logo_prompts"], str) else list(o["logo_prompts"])
    if "logos" in layers and not prompts:
        raise ValueError("logo_prompts must not be empty")

    stills = opts.get("stills")
    if stills is not None:
        media = {"media_type": "stills", "width": None, "height": None, "duration": None}
        frames = [(float(ts), _ocr._load_image(p)) for ts, p in _ocr._iter_stills(stills)]
    else:
        media, frames = _ocr._read_media(path, fps, int(o["max_frames"] or 0), int(o["timeout"]))

    torch = _torch(int(o["cpu_threads"]))
    objects = _detector(o["object_model"], "closed", o) if "objects" in layers else None
    logos = _detector(o["logo_model"], "open", o) if "logos" in layers else None
    refs = _brand_refs(o) if logos and o["brand_match"] and o["brand_refs_dir"] else None
    brands = _brand_names(o, refs)
    ocr = _ocr._pipeline("latin", _ocr.DEFAULTS)[0] if logos and o["logo_ocr"] else None

    per_frame = []
    for ts, bgr in frames:
        img = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
        entry = {"timestamp": round(ts, 3)}
        if objects:
            entry["objects"] = _detect(torch, objects, img, None, float(o["object_threshold"]))
        if logos:
            found = _detect(torch, logos, img, prompts, float(o["logo_threshold"]))
            found = _nms(found, 0.5)[: int(o["max_logos_per_frame"])]
            for logo in found:
                logo.update(_identify(torch, img, logo["box"], ocr, refs, brands, o))
            if ocr is not None and brands and o["wordmarks"]:
                found = _merge_wordmarks(found, _wordmarks(ocr, bgr, brands))
            entry["logos"] = found
        per_frame.append(entry)

    models = {}
    if objects:
        models["objects"] = objects["info"]
    if logos:
        models["logos"] = logos["info"]
    if refs:
        models["brand_embeddings"] = refs["info"]
    if ocr:
        models["logo_ocr"] = {"tool": _ocr.TOOL, "tool_version": _ocr._version("rapidocr")}
    label_frames = Counter(lbl for f in per_frame for lbl in {d["label"] for d in f.get("objects", [])})
    return {
        "tool": TOOL,
        "tool_version": _version("transformers"),
        "torch_version": _version("torch"),
        "model": "; ".join(f"{k}={v['id']}@{(v['revision'] or '')[:12]}" for k, v in models.items() if "id" in v),
        "models": models,
        **media,
        "sample_fps": fps if media["media_type"] == "video" else None,
        "frame_count": len(per_frame),
        "layers": list(layers),
        "object_labels": dict(label_frames.most_common()),
        "logo_frames": sum(1 for f in per_frame if f.get("logos")),
        "brands": _brand_timeline(per_frame),
        "has_objects": bool(label_frames),
        "has_logos": any(f.get("logos") for f in per_frame),
        "frames": per_frame,
    }


class ObjectDetectionEnricher(Enricher):
    """Runs `analyze` on every image and video in the item and stores the result on the Media under
    `object_detection` (or `object_detection_error`)."""

    def enrich(self, to_enrich: Metadata) -> None:
        opts = {k: getattr(self, k, None) for k in DEFAULTS}
        for media in to_enrich.media:
            if not (media.is_video() or media.is_image()):
                continue
            try:
                result = analyze(media.filename, **opts)
            except (EnricherInputError, EnricherServiceError) as e:
                logger.warning(f"object detection failed for {media.filename}: {type(e).__name__} {e.reason}: {e}")
                media.set("object_detection_error", {"type": type(e).__name__, "reason": e.reason, "message": str(e)})
                continue
            media.set("object_detection", result)
            logger.debug(f"objects {media.filename}: {result['object_labels']} logos in {result['logo_frames']} frames")


def _layers(value) -> tuple:
    layers = (value,) if isinstance(value, str) else tuple(value)
    bad = [x for x in layers if x not in LAYERS]
    if bad or not layers:
        raise ValueError(f"layers must be a non-empty subset of {LAYERS}, got {list(layers)}")
    return layers


def _version(dist: str):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


# ---------- models ----------


def _torch(threads: int):
    try:
        import torch
    except ImportError as e:
        raise EnricherServiceError(
            "torch is not installed (pip install -r object_detection_enricher/requirements.txt)", reason="tool_missing"
        ) from e
    if threads > 0 and torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
    return torch


def _load(kind: str, model_id: str, o: dict):
    """(model, processor) cached per process. kind: closed | open | embed."""
    key = (kind, model_id, o["model_dir"])
    with _MODELS_LOCK:
        if key in _MODELS:
            return _MODELS[key]
        try:
            import transformers
        except ImportError as e:
            raise EnricherServiceError("transformers is not installed", reason="tool_missing") from e
        auto = {
            "closed": (transformers.AutoModelForObjectDetection, transformers.AutoImageProcessor),
            "open": (transformers.AutoModelForZeroShotObjectDetection, transformers.AutoProcessor),
            "embed": (transformers.AutoModel, transformers.AutoProcessor),
        }[kind]
        kw = {"cache_dir": o["model_dir"]} if o["model_dir"] else {}
        try:
            model = auto[0].from_pretrained(model_id, **kw).eval()
            proc = auto[1].from_pretrained(model_id, **kw)
        except MemoryError as e:
            raise EnricherServiceError(f"out of memory loading {model_id}", reason="out_of_memory") from e
        except (OSError, ValueError, KeyError, ImportError, RuntimeError) as e:
            # HF raises OSError for unknown ids, offline cache misses and HTTP failures, ValueError/KeyError for
            # checkpoints of the wrong head type; none of these produce a usable model
            raise EnricherServiceError(
                f"model {model_id!r} unavailable: {_ocr._root_cause(e)}", reason="model_unavailable"
            ) from e
        _MODELS[key] = (model, proc)
        return _MODELS[key]


def _detector(model_id: str, kind: str, o: dict) -> dict:
    model, proc = _load(kind, model_id, o)
    info = {
        "id": model_id,
        "revision": getattr(model.config, "_commit_hash", None),
        "architecture": model.config.model_type,
        "license_note": "see model card",
    }
    return {"kind": kind, "model": model, "proc": proc, "info": info}


def _infer(torch, fn, what: str):
    try:
        with torch.inference_mode():
            return fn()
    except MemoryError as e:
        raise EnricherServiceError(f"out of memory in {what}", reason="out_of_memory") from e
    except ImportError as e:
        # processors import optional backends lazily (OWLv2 needs scipy)
        raise EnricherServiceError(f"{what}: {e}".strip(), reason="tool_missing") from e
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "can't allocate memory" in str(e).lower():
            raise EnricherServiceError(f"out of memory in {what}: {e}", reason="out_of_memory") from e
        raise EnricherServiceError(f"{what} failed: {_ocr._root_cause(e)}", reason="detection_failed") from e


def _detect(torch, det: dict, img: Image.Image, prompts, threshold: float) -> list:
    model, proc = det["model"], det["proc"]
    w, h = img.size
    arch = model.config.model_type

    def run():
        if prompts is None:
            out = model(**proc(images=img, return_tensors="pt"))
            r = proc.post_process_object_detection(out, threshold=threshold, target_sizes=[(h, w)])[0]
            labels = [model.config.id2label[int(i)] for i in r["labels"]]
            return r, labels, (w, h)
        inputs = proc(text=[prompts], images=img, return_tensors="pt")
        out = model(**inputs)
        if arch == "owlv2":
            # OWLv2 pads the image to a square before resizing; boxes are relative to that square
            s = max(w, h)
            r = proc.post_process_grounded_object_detection(
                out, threshold=threshold, target_sizes=[(s, s)], text_labels=[prompts]
            )[0]
        else:
            r = proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=threshold, text_threshold=threshold, target_sizes=[(h, w)]
            )[0]
        labels = [str(x) for x in (r.get("text_labels") or r["labels"])]
        return r, labels, (w, h)

    r, labels, (w, h) = _infer(torch, run, f"{arch} detection")
    found = []
    for score, label, box in zip(r["scores"].tolist(), labels, r["boxes"].tolist()):
        b = _norm_box(box, w, h)
        if b[2] > b[0] and b[3] > b[1]:
            found.append({"label": label, "score": round(float(score), 4), "box": b})
    return sorted(found, key=lambda d: -d["score"])


def _norm_box(box, w: int, h: int) -> list:
    x0, y0, x1, y1 = box
    return [round(min(max(x0 / w, 0.0), 1.0), 4), round(min(max(y0 / h, 0.0), 1.0), 4),
            round(min(max(x1 / w, 0.0), 1.0), 4), round(min(max(y1 / h, 0.0), 1.0), 4)]  # fmt: skip


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _nms(dets: list, iou: float) -> list:
    kept = []
    for d in dets:
        if all(_iou(d["box"], k["box"]) < iou for k in kept):
            kept.append(d)
    return kept


# ---------- brand identity ----------


def _brand_names(o: dict, refs) -> list:
    names = [o["brands"]] if isinstance(o["brands"], str) else list(o["brands"] or ())
    if refs:
        names += [b for b in refs["brands"] if b not in names]
    return names


def _brand_refs(o: dict) -> dict:
    """SigLIP 2 unit vectors of every reference logo under brand_refs_dir/<brand>/<image>."""
    root = o["brand_refs_dir"]
    if not os.path.isdir(root):
        raise ValueError(f"brand_refs_dir is not a directory: {root}")
    files = []
    for brand in sorted(os.listdir(root)):
        d = os.path.join(root, brand)
        if os.path.isdir(d):
            files += [(brand, os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.lower().endswith(IMAGE_EXTS)]
    key = (o["embed_model"], root, tuple((f, os.path.getmtime(f)) for _b, f in files))
    if key in _REFS:
        return _REFS[key]
    model, proc = _load("embed", o["embed_model"], o)
    torch = _torch(0)
    images = []
    for _b, f in files:
        try:
            with Image.open(f) as im:
                images.append(_on_white(im))
        except OSError as e:
            raise ValueError(f"unreadable reference logo {f}: {e}") from e
    vecs = _embed(torch, model, proc, images) if images else np.zeros((0, 1), dtype=np.float32)
    info = {"id": o["embed_model"], "revision": getattr(model.config, "_commit_hash", None), "references": len(files)}
    _REFS[key] = {
        "brands": sorted({b for b, _f in files}),
        "labels": [b for b, _f in files],
        "vecs": vecs,
        "info": info,
    }
    return _REFS[key]


def _on_white(im: Image.Image) -> Image.Image:
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, "white")
        return Image.alpha_composite(bg, im).convert("RGB")
    return im.convert("RGB")


def _embed(torch, model, proc, images: list) -> np.ndarray:
    def run():
        out = model.get_image_features(**proc(images=images, return_tensors="pt"))
        feats = out if hasattr(out, "norm") else out.pooler_output
        return (feats / feats.norm(dim=-1, keepdim=True)).float().numpy()

    return _infer(torch, run, "SigLIP 2 embedding")


def _crop(img: Image.Image, box: list, pad: float = 0.1) -> Image.Image:
    w, h = img.size
    bw, bh = (box[2] - box[0]) * w, (box[3] - box[1]) * h
    x0, y0 = max(0, box[0] * w - pad * bw), max(0, box[1] * h - pad * bh)
    x1, y1 = min(w, box[2] * w + pad * bw), min(h, box[3] * h + pad * bh)
    return img.crop((int(x0), int(y0), max(int(x0) + 1, int(x1)), max(int(y0) + 1, int(y1))))


def _canon(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", s.lower().replace("’", "'"))


def _text_brand(text: str, brands: list):
    """(brand, similarity) when the OCR text in a logo box reads as a brand name, else (None, 0)."""
    t = _canon(text)
    if len(t) < 2:
        return None, 0.0
    best, best_s = None, 0.0
    for b in brands:
        c = _canon(b)
        if not c:
            continue
        s = 1.0 if (c in t and len(c) >= 3) else difflib.SequenceMatcher(None, t, c).ratio()
        if s > best_s:
            best, best_s = b, s
    return (best, round(best_s, 3)) if best_s >= 0.8 else (None, 0.0)


def _identify(torch, img: Image.Image, box: list, ocr, refs, brands: list, o: dict) -> dict:
    crop = _crop(img, box)
    out = {"brand": None, "brand_score": None, "brand_source": None, "text": None}
    if ocr is not None and min(crop.size) >= 8:
        lines = _ocr._ocr_frame(ocr, "latin", np.ascontiguousarray(np.asarray(crop)[:, :, ::-1]), 0.6)
        text = " ".join(line["text"] for line in lines).strip()
        if text:
            out["text"] = text
            brand, s = _text_brand(text, brands)
            if brand:
                out.update(brand=brand, brand_score=s, brand_source="ocr")
    if out["brand"] is None and refs is not None and len(refs["labels"]):
        model, proc = _load("embed", o["embed_model"], o)
        v = _embed(torch, model, proc, [crop])[0]
        sims = refs["vecs"] @ v
        i = int(np.argmax(sims))
        if float(sims[i]) >= float(o["brand_match_threshold"]):
            out.update(brand=refs["labels"][i], brand_score=round(float(sims[i]), 4), brand_source="siglip2")
    return out


def _wordmarks(ocr, bgr: np.ndarray, brands: list) -> list:
    """Full-frame OCR lines that read as a brand name: wordmark logos the open-vocabulary detector missed."""
    h, w = bgr.shape[:2]
    found = []
    for line in _ocr._ocr_frame(ocr, "latin", bgr, 0.6):
        brand, s = _text_brand(line["text"], brands)
        # a caption that mentions the brand ("Samsung completely touchscreen tablet") is not a wordmark
        if brand and len(_canon(line["text"])) <= len(_canon(brand)) + 6:
            x0, y0, x1, y1 = line["box"]
            found.append({"label": "wordmark", "score": line["confidence"], "box": _norm_box((x0, y0, x1, y1), w, h),
                          "brand": brand, "brand_score": s, "brand_source": "ocr", "text": line["text"]})  # fmt: skip
    return found


def _merge_wordmarks(logos: list, words: list) -> list:
    """A wordmark inside a detected logo box names that logo; any other wordmark becomes a logo of its own."""
    out = list(logos)
    for wm in words:
        hit = next((lg for lg in logos if _iou(lg["box"], wm["box"]) > 0.3 or _inside(wm["box"], lg["box"])), None)
        if hit is None:
            out.append(wm)
        elif hit["brand"] is None:
            hit.update(brand=wm["brand"], brand_score=wm["brand_score"], brand_source="ocr", text=wm["text"])
    return out


def _inside(a, b) -> bool:
    return a[0] >= b[0] - 0.01 and a[1] >= b[1] - 0.01 and a[2] <= b[2] + 0.01 and a[3] <= b[3] + 0.01


def _brand_timeline(frames: list) -> list:
    seen = defaultdict(lambda: {"first_seen": None, "last_seen": None, "frames": 0, "max_score": 0.0, "sources": set()})
    for f in frames:
        for logo in f.get("logos", []):
            if not logo.get("brand"):
                continue
            s = seen[logo["brand"]]
            if s["first_seen"] is None:
                s["first_seen"] = f["timestamp"]
            if s["last_seen"] != f["timestamp"]:
                s["frames"] += 1
            s["last_seen"] = f["timestamp"]
            s["max_score"] = max(s["max_score"], logo["brand_score"] or 0.0)
            s["sources"].add(logo["brand_source"])
    return [{"brand": b, **{**v, "sources": sorted(v["sources"])}} for b, v in sorted(seen.items())]
