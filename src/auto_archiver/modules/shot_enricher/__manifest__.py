{
    "name": "Shot Enricher",
    "type": ["enricher"],
    "requires_setup": False,
    "dependencies": {"python": ["loguru", "scenedetect", "cv2"], "bin": ["ffmpeg", "ffprobe"]},
    "configs": {
        "detector": {
            "default": "adaptive",
            "choices": ["adaptive", "content"],
            "help": "PySceneDetect detector: AdaptiveDetector (rolling-average content change, fewer false cuts on camera motion) or ContentDetector (fixed threshold).",
        },
        "adaptive_threshold": {
            "default": 3.0,
            "type": "float",
            "help": "AdaptiveDetector: ratio of a frame's content change to its neighbours' average that marks a cut.",
        },
        "min_content_val": {
            "default": 15.0,
            "type": "float",
            "help": "AdaptiveDetector: minimum content change a frame needs to be a cut.",
        },
        "content_threshold": {
            "default": 27.0,
            "type": "float",
            "help": "ContentDetector: content-change threshold for a cut.",
        },
        "min_scene_len": {
            "default": 0.25,
            "type": "float",
            "help": "shortest shot in seconds; cuts closer than this are dropped (PySceneDetect's own default is 15 frames, too long for fast-cut ads).",
        },
        "downscale": {
            "default": None,
            "type": "int",
            "help": "decode downscale factor for detection; empty lets PySceneDetect choose from the frame width.",
        },
        "keyframes": {
            "default": True,
            "type": "bool",
            "help": "save the middle frame of each shot as a JPEG next to the video.",
        },
        "keyframe_quality": {"default": 90, "type": "int", "help": "JPEG quality of the keyframes (0-100)."},
        "black_detection": {
            "default": True,
            "type": "bool",
            "help": "run ffmpeg blackdetect to report black and faded-to-black segments.",
        },
        "black_min_duration": {
            "default": 0.1,
            "type": "float",
            "help": "blackdetect d: shortest black segment in seconds.",
        },
        "black_pixel_threshold": {
            "default": 0.10,
            "type": "float",
            "help": "blackdetect pix_th: luminance (0-1) at or below which a pixel counts as black.",
        },
        "black_picture_ratio": {
            "default": 0.98,
            "type": "float",
            "help": "blackdetect pic_th: share of black pixels for a frame to count as black.",
        },
        "black_shot_ratio": {
            "default": 0.9,
            "type": "float",
            "help": "a shot is flagged black when at least this share of it is inside black segments.",
        },
        "timeout": {
            "default": 600,
            "type": "int",
            "help": "seconds before shot detection or blackdetect is stopped (raises EnricherServiceError reason=timeout).",
        },
        "probe_timeout": {"default": 60, "type": "int", "help": "seconds before ffprobe is stopped."},
    },
    "description": """
    Detects the shots (scenes) of each video with PySceneDetect (BSD-3-Clause) and ffmpeg blackdetect.

    ### Features
    - Shot list with start, end and duration in seconds and frames, cut count, cuts per minute, mean and median shot length.
    - `AdaptiveDetector` by default, `ContentDetector` as an option; thresholds and minimum shot length are configurable.
    - One keyframe per shot (its middle frame) saved as JPEG next to the video and added as `shot_keyframes` Media,
      so `pdq_hash_enricher` can hash them when it runs after this enricher.
    - Black / fade-to-black segments from ffmpeg `blackdetect`; shots that are almost all black are flagged.
    - `analyze(path, **opts)` is a pure function returning a JSON-serialisable dict (`tool`, `tool_version`, `model`=None, ...)
      for callers that run it per stored file. Typed failures: `EnricherInputError` (bad media, don't retry) and
      `EnricherServiceError` (tool missing, crash, timeout, out of memory; may be retried).

    ### Notes
    - Runs in-process on CPU (OpenCV decoding); no model server or docker-compose service is needed.
    - Requires `ffmpeg` and `ffprobe` on PATH (run as subprocesses).
    - A video with no cuts is a valid result (`single_shot`=True, `empty_reason`="no_cuts"); a still image gives an empty
      result with `empty_reason`="still_image".
    - For variable-frame-rate input the times are taken from timestamps but the frame numbers are estimates (`frames_approximate`).
    """,
}
