{
    "name": "Soundtrack Enricher",
    "type": ["enricher"],
    "requires_setup": False,
    "dependencies": {
        "python": ["loguru", "numpy", "torch", "demucs", "beat_this", "librosa", "soundfile", "transformers"],
        "bin": ["ffmpeg", "ffprobe", "fpcalc"],
    },
    "configs": {
        "device": {"default": "cpu", "help": "cpu, cuda (or cuda:N) or auto; CPU always works, a GPU is optional"},
        "cpu_threads": {"default": 0, "type": "int", "help": "torch intra-op threads, 0 keeps the default"},
        "separation_model": {"default": "htdemucs", "help": "Demucs v4 model splitting vocals from the music bed"},
        "separation_shifts": {
            "default": 0,
            "type": "int",
            "help": "Demucs random shifts; each one costs a full extra pass",
        },
        "clap_model": {"default": "laion/larger_clap_music_and_speech", "help": "Hugging Face id of a CLAP checkpoint"},
        "clap_window_s": {"default": 10.0, "type": "float", "help": "CLAP window length in seconds (half-overlapping)"},
        "beat_model": {"default": "final0", "help": "beat_this checkpoint"},
        "music_threshold_db": {
            "default": -42.0,
            "type": "float",
            "help": "music-stem level (dBFS per 0.5 s) that counts as music",
        },
        "voice_threshold_db": {
            "default": -40.0,
            "type": "float",
            "help": "vocal-stem level (dBFS per 5 s window) checked for singing vs speech",
        },
        "min_music_s": {"default": 1.0, "type": "float", "help": "drop music ranges shorter than this"},
        "max_gap_s": {"default": 1.0, "type": "float", "help": "join music ranges separated by at most this"},
        "max_seconds": {"default": 600, "type": "int", "help": "analyse at most this many seconds, 0 means no limit"},
        "acoustid_key": {
            "default": None,
            "help": "AcoustID application key; unset turns naming off (default $ACOUSTID_API_KEY)",
        },
        "acoustid_url": {
            "default": "https://api.acoustid.org/v2/lookup",
            "help": "AcoustID lookup endpoint, e.g. a self-hosted acoustid-server",
        },
        "model_dir": {
            "default": None,
            "help": "model cache directory (default $SOUNDTRACK_MODEL_DIR, else torch/HF defaults)",
        },
        "keep_stems_dir": {"default": None, "help": "write music.wav and vocals.wav here instead of discarding them"},
        "timeout": {"default": 900, "type": "int", "help": "seconds allowed for each ffmpeg/ffprobe/fpcalc call"},
    },
    "description": """
    Finds and describes the music under an ad's voice-over, so ads sharing a track can be grouped even when the track has no name.

    ### Features
    - Separation: Demucs v4 htdemucs (MIT) splits vocals from drums/bass/other; the music bed is everything but vocals.
    - Music presence: time ranges where the music bed is above `music_threshold_db`, and the share of the file.
    - Fingerprint: Chromaprint (fpcalc, LGPL-2.1+) of the music bed, same format as audio_enricher, for same-track matching.
    - Tempo: beat_this (MIT, CPJKU 2024). Key: Krumhansl-Schmuckler profile on librosa chroma (ISC).
    - Genre, mood, sung vs spoken, and a 512-d embedding: LAION CLAP larger_clap_music_and_speech (Apache-2.0) zero-shot.
    - Naming (optional): AcoustID / MusicBrainz lookup of the music-bed fingerprint when `acoustid_key` is set.
      The hosted AcoustID API is free for non-commercial use only; commercial use needs an AcoustID.biz plan or a
      self-hosted acoustid-server (MIT) with the CC BY-SA 3.0 data dumps (`acoustid_url`).

    No Essentia models (non-commercial licences) and no AGPL code are used.

    ### Usage
    `analyze(path, **opts)` returns a JSON-serialisable dict; `EnricherInputError` for unreadable media,
    `EnricherServiceError` for tool/model failures (retry may help).
    ### Setup
    `pip install --extra-index-url https://download.pytorch.org/whl/cpu -r soundtrack_enricher/requirements.txt`
    and fpcalc (`apt install libchromaprint-tools`). Models download on first use (htdemucs ~84 MB, CLAP ~780 MB,
    beat_this ~80 MB).
    """,
}
