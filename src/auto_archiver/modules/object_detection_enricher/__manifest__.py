{
    "name": "Object Detection Enricher",
    "type": ["enricher"],
    "requires_setup": False,
    "dependencies": {
        "python": ["loguru", "PIL", "numpy", "torch", "transformers", "rapidocr", "onnxruntime"],
        "bin": ["ffmpeg", "ffprobe"],
    },
    "configs": {
        "layers": {"default": ["objects", "logos"], "help": "what to detect: any of objects, logos"},
        "fps": {"default": 1.0, "type": "float", "help": "frames sampled per second of video"},
        "max_frames": {"default": 0, "type": "int", "help": "stop sampling after this many frames, 0 means no limit"},
        "object_model": {
            "default": "PekingU/rtdetr_v2_r18vd",
            "help": "Hugging Face id of a closed-set detector (AutoModelForObjectDetection: RT-DETR(v2), D-FINE, DETR ...)",
        },
        "object_threshold": {"default": 0.5, "type": "float", "help": "drop object boxes below this score (0..1)"},
        "logo_model": {
            "default": "google/owlv2-base-patch16-ensemble",
            "help": "Hugging Face id of an open-vocabulary detector (OWLv2 or Grounding DINO) prompted with logo_prompts",
        },
        "logo_prompts": {"default": ["logo"], "help": "text prompts for the logo detector"},
        "logo_threshold": {"default": 0.2, "type": "float", "help": "drop logo boxes below this score (0..1)"},
        "max_logos_per_frame": {"default": 10, "type": "int", "help": "keep at most this many logo boxes per frame"},
        "brands": {"default": [], "help": "brand names to recognise in the OCR text inside logo boxes"},
        "brand_refs_dir": {
            "default": None,
            "help": "folder of reference logos, <brand>/<image>; logo crops are matched to them with SigLIP 2 (default $OD_BRAND_REFS_DIR)",
        },
        "embed_model": {
            "default": "google/siglip2-base-patch16-224",
            "help": "image embedding model for brand matching",
        },
        "brand_match_threshold": {"default": 0.8, "type": "float", "help": "min cosine similarity to a reference logo"},
        "logo_ocr": {"default": True, "type": "bool", "help": "read the text inside each logo box with RapidOCR"},
        "wordmarks": {
            "default": True,
            "type": "bool",
            "help": "also OCR the whole frame and add text lines that read as one of `brands` as wordmark logos",
        },
        "cpu_threads": {"default": 0, "type": "int", "help": "torch intra-op threads, 0 keeps the default"},
        "device": {"default": "auto", "help": "auto (GPU when torch sees one), cpu, or cuda[:n] (fails without a GPU)"},
        "model_dir": {"default": None, "help": "Hugging Face cache directory (default $OD_MODEL_DIR, else $HF_HOME)"},
        "timeout": {"default": 600, "type": "int", "help": "seconds allowed for ffprobe/ffmpeg frame extraction"},
    },
    "description": """
    Finds objects and logos in images and video frames for highlighting, on CPU, with Hugging Face transformers.

    ### Features
    - Objects: a closed-set COCO detector, RT-DETRv2-R18 by default (Apache-2.0).
    - Logos: an open-vocabulary detector prompted with "logo", OWLv2 base by default (Apache-2.0).
    - Brand per logo box: the text inside the box (RapidOCR, as in ocr_enricher) fuzzy-matched to `brands`
      and the reference-logo folder names, else the nearest SigLIP 2 embedding of a reference logo above
      `brand_match_threshold`. No brand list is built in; pass your own.
    - Videos are sampled exactly like ocr_enricher (ffmpeg, source timestamps), so OCR text boxes, objects and
      logos of the same frame line up.
    - Per frame: `timestamp`, `objects` and `logos`, each `{label, score, box}` with box = [x0, y0, x1, y1]
      normalised 0..1; logos also have `brand`, `brand_score`, `brand_source` (ocr | siglip2) and `text`.
    - `models` records each model id, Hub revision and architecture; `tool_version`/`torch_version` the runtime.
    - Failures are typed: `EnricherInputError` (bad media, don't retry) and `EnricherServiceError` with reason
      `model_unavailable`, `tool_missing`, `out_of_memory`, `detection_failed`, `ffmpeg_*`. A frame with nothing
      detected is not an error.

    ### Setup
    - `pip install --extra-index-url https://download.pytorch.org/whl/cpu -r src/auto_archiver/modules/object_detection_enricher/requirements.txt`
    - Models download from the Hugging Face Hub on first use into `model_dir` / `$OD_MODEL_DIR` / `$HF_HOME`.
      Bake them into the image and set `HF_HUB_OFFLINE=1` for offline workers.
    - Requires `ffmpeg` and `ffprobe` on the PATH.
    """,
}
