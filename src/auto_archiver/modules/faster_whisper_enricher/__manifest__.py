{
    "name": "Faster-Whisper Transcript Enricher",
    "type": ["enricher"],
    "requires_setup": True,
    "dependencies": {
        "python": ["loguru", "numpy", "faster_whisper"],
        "bin": ["ffmpeg", "ffprobe"],
    },
    "configs": {
        "model": {
            "default": "small",
            "help": "faster-whisper model size or CTranslate2 model id/path, e.g. small, distil-large-v3 (English only), Systran/faster-whisper-medium.",
        },
        "device": {"default": "cpu", "help": "cpu, cuda or auto."},
        "compute_type": {"default": "int8", "help": "CTranslate2 compute type, e.g. int8, int8_float16, float16."},
        "cpu_threads": {"default": 0, "type": "int", "help": "CPU threads per transcription, 0 = CTranslate2 default."},
        "language": {"default": None, "help": "Force a language code (e.g. fr); empty = detect."},
        "beam_size": {"default": 5, "type": "int", "help": "Beam size for decoding."},
        "word_timestamps": {"default": False, "type": "bool", "help": "Add per-word timestamps to each segment."},
        "download_root": {"default": None, "help": "Directory for downloaded models (default: Hugging Face cache)."},
        "local_files_only": {
            "default": False,
            "type": "bool",
            "help": "Never download the model; fail with EnricherServiceError if it is not cached.",
        },
        "decode_timeout": {
            "default": 600,
            "type": "int",
            "help": "Seconds allowed for ffprobe/ffmpeg to read and decode the audio track.",
        },
    },
    "description": """
    Transcribes speech in video and audio files and detects the spoken language with faster-whisper
    (https://github.com/SYSTRAN/faster-whisper, MIT; CTranslate2 Whisper models, MIT).

    ### Features
    - Runs in-process on CPU by default (int8 `small` model), no model server needed.
    - Silero VAD (bundled with faster-whisper) removes non-speech audio before Whisper sees it, so music-only
      and silent media return empty text with `speech: false` instead of hallucinated captions.
    - Stores the result under the `transcript` key of each video/audio `Media`: language and probability,
      segments (start, end, text, avg_logprob, no_speech_prob, optional words), full text, speech duration,
      `tool`, `tool_version` and `model`.
    - `analyze(path, **opts)` is a pure function callable per file; it raises `EnricherInputError` for media that
      can't be read and `EnricherServiceError` for tool/model failures (see `auto_archiver.utils.enricher_errors`).

    ### Notes
    - Install with the `transcription` extra (`pip install auto-archiver[transcription]`); needs `ffmpeg`/`ffprobe`.
    - The model is downloaded from the Hugging Face Hub on first use and cached per worker process (~0.8 GB RSS for `small` int8).
      Set `download_root` + `local_files_only` to run offline from a pre-baked model directory.
    - `distil-large-v3` is English-only; keep `small` (or larger multilingual models) for non-English ads.
    - `analyze` is blocking and CPU bound; from async code run it in a thread or process pool.
    """,
}
