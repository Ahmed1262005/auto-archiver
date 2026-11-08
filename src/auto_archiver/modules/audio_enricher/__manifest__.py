{
    "name": "Audio Enricher",
    "type": ["enricher"],
    "requires_setup": False,
    "dependencies": {
        "python": ["loguru", "numpy"],
        "bin": ["ffmpeg", "ffprobe", "fpcalc"],
    },
    "configs": {
        "silence_noise_db": {
            "default": -50.0,
            "type": "float",
            "help": "silencedetect noise floor in dB; audio below it counts as silence",
        },
        "silence_min_duration": {
            "default": 0.5,
            "type": "float",
            "help": "shortest silence (seconds) reported as a segment",
        },
        "black_min_duration": {
            "default": 0.1,
            "type": "float",
            "help": "shortest black stretch (seconds) reported as a segment",
        },
        "black_pixel_threshold": {
            "default": 0.10,
            "type": "float",
            "help": "blackdetect pix_th: luma below this fraction of the range is a black pixel",
        },
        "black_picture_threshold": {
            "default": 0.98,
            "type": "float",
            "help": "blackdetect pic_th: share of black pixels for a frame to count as black",
        },
        "fingerprint_max_seconds": {
            "default": 0,
            "type": "int",
            "help": "fingerprint at most this many seconds of audio (fpcalc -length); 0 means the whole stream",
        },
        "timeout": {
            "default": 600,
            "type": "int",
            "help": "seconds each ffprobe/ffmpeg/fpcalc run may take before it is killed",
        },
    },
    "description": """
    Audio analysis and audio fingerprints for video and audio files, using ffmpeg and Chromaprint as subprocesses.

    ### Features
    - EBU R128 integrated loudness (LUFS), loudness range (LU) and true peak (dBTP) with ffmpeg's `ebur128` filter.
    - Silence segments and silent fraction (`silencedetect`); black-frame segments and black fraction (`blackdetect`).
    - Chromaprint audio fingerprint (`fpcalc -raw`, algorithm 2) and its duration, stored on each `Media` as `audio`.
    - `fingerprint_similarity(fp_a, fp_b)` compares two raw fingerprints at every offset (bit-error rate over a
      sliding window), so a cut-down or trimmed edit matches its longer version at the right offset.
    - `analyze(path)` is a pure function that callers can run per stored file; it raises
      `EnricherInputError` (bad media, don't retry) or `EnricherServiceError` (tool failed, retry may help).

    ### Notes
    - Needs `ffmpeg`/`ffprobe` and `fpcalc` (Debian/Ubuntu package `libchromaprint-tools`, Chromaprint is LGPL-2.1+)
      on PATH. Both run as subprocesses; no GPL/LGPL code is imported in-process. No model server is needed.
    - Valid media with no audio stream, or audio that is entirely silent, is not an error: `has_audio` is false,
      `empty_reason` says why and the fingerprint is empty.
    """,
}
