{
    "name": "OCR Enricher",
    "type": ["enricher"],
    "requires_setup": False,
    "dependencies": {
        "python": ["loguru", "PIL", "numpy", "paddleocr", "paddle"],
        "bin": ["ffmpeg", "ffprobe"],
    },
    "configs": {
        "languages": {
            "default": ["latin"],
            "help": "recognisers to run: any of latin, arabic, chinese, japanese, korean (or a raw PaddleOCR lang code such as 'fr' or 'ar'). Each extra language runs one more PaddleOCR pipeline per frame.",
        },
        "fps": {"default": 1.0, "type": "float", "help": "frames sampled per second of video"},
        "min_confidence": {
            "default": 0.6,
            "type": "float",
            "help": "drop recognised lines below this score (0..1)",
        },
        "max_frames": {"default": 0, "type": "int", "help": "stop sampling after this many frames, 0 means no limit"},
        "device": {"default": "cpu", "help": "PaddlePaddle device, e.g. cpu or gpu:0"},
        "cpu_threads": {"default": 0, "type": "int", "help": "CPU threads per pipeline, 0 keeps the PaddleOCR default"},
        "ocr_version": {
            "default": None,
            "help": "force a PP-OCR version (PP-OCRv5, PP-OCRv6); default picks per language",
        },
        "timeout": {"default": 600, "type": "int", "help": "seconds allowed for ffprobe/ffmpeg frame extraction"},
    },
    "description": """
    Reads on-screen text (captions, supers, prices, URLs, end cards) from images and video frames with
    [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0) and its PP-OCR multilingual models.

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
    - `pip install -r src/auto_archiver/modules/ocr_enricher/requirements.txt` (pinned paddleocr + paddlepaddle CPU).
    - Models download on first use (HuggingFace, then other PaddleX hosters) into `~/.paddlex/official_models`
      (`PADDLE_PDX_CACHE_HOME` moves it). Bake them into the image for offline workers.
    - Requires `ffmpeg` and `ffprobe` on the PATH.
    """,
}
