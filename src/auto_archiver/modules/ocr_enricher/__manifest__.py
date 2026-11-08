{
    "name": "OCR Enricher",
    "type": ["enricher"],
    "requires_setup": False,
    "dependencies": {
        "python": ["loguru", "PIL", "numpy", "rapidocr", "onnxruntime"],
        "bin": ["ffmpeg", "ffprobe"],
    },
    "configs": {
        "languages": {
            "default": ["latin"],
            "help": "recognisers to run: any of latin, arabic, chinese, japanese, korean (or a raw RapidOCR lang code such as 'cyrillic' or 'th'). Latin, Chinese and Japanese share one PP-OCRv6 model; each other language runs one more pipeline per frame.",
        },
        "fps": {"default": 1.0, "type": "float", "help": "frames sampled per second of video"},
        "min_confidence": {
            "default": 0.6,
            "type": "float",
            "help": "drop recognised lines below this score (0..1)",
        },
        "max_frames": {"default": 0, "type": "int", "help": "stop sampling after this many frames, 0 means no limit"},
        "model_type": {
            "default": "small",
            "help": "PP-OCRv6 model size: tiny, small or medium (PP-OCRv5 languages use mobile)",
        },
        "cpu_threads": {"default": 0, "type": "int", "help": "ONNX Runtime threads per pipeline, 0 keeps the default"},
        "model_dir": {
            "default": None,
            "help": "model cache directory (default $OCR_MODEL_DIR, else inside the rapidocr package)",
        },
        "ocr_version": {
            "default": None,
            "help": "force a PP-OCR version (PP-OCRv5, PP-OCRv6); default picks per language",
        },
        "timeout": {"default": 600, "type": "int", "help": "seconds allowed for ffprobe/ffmpeg frame extraction"},
    },
    "description": """
    Reads on-screen text (captions, supers, prices, URLs, end cards) from images and video frames with
    [RapidOCR](https://github.com/RapidAI/RapidOCR) (Apache-2.0), which runs PaddleOCR's PP-OCR models on ONNX Runtime.

    ### Features
    - Videos are sampled with ffmpeg at `fps` (default 1 frame/s); images are read directly.
    - Per frame: text lines with box, polygon, confidence, recogniser language and Unicode script.
    - A de-duplicated timeline: each distinct line with `first_seen`, `last_seen`, `max_confidence` and its box.
    - `full_text`, `scripts`, `has_text` (+ `empty_reason`), `tool`, `tool_version`, `model`.
    - `analyze(path, **opts)` is a pure function that can be called without the archiver; `enrich` stores its
      result on each image/video `Media` under `ocr` (or `ocr_error` with the failure type and reason).
    - Failures are typed: `EnricherInputError` (bad media, don't retry) and `EnricherServiceError`
      (ffmpeg/OCR/model download failed, retry may help). Valid media without text is not an error.

    ### Setup
    - Runs in-process on CPU; no model server is needed.
    - `pip install -r src/auto_archiver/modules/ocr_enricher/requirements.txt` (pinned rapidocr + onnxruntime CPU).
    - Models (ONNX) download on first use into `model_dir` / `$OCR_MODEL_DIR`, else the rapidocr package's
      `models/` folder. Bake them into the image for offline workers.
    - Requires `ffmpeg` and `ffprobe` on the PATH.
    """,
}
