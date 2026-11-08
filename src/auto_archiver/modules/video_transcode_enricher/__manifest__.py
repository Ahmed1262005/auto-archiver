{
    "name": "Video Transcode Enricher",
    "type": ["enricher"],
    "requires_setup": False,
    "dependencies": {"python": ["loguru", "ffmpeg"], "bin": ["ffmpeg", "ffprobe"]},
    "configs": {
        "heights": {
            "default": [1080, 720, 480, 360],
            "type": "json_loader",
            "help": "rendition ladder, by the short side of the displayed frame (1080 = 1080p landscape or 1080x1920 portrait). Never upscales.",
        },
        "crf": {"default": 23, "type": "int", "help": "libx264 constant rate factor"},
        "preset": {"default": "medium", "help": "libx264 preset"},
        "audio_bitrate": {"default": "128k", "help": "AAC bitrate"},
        "min_height": {"default": 240, "type": "int", "help": "videos with a shorter short side are not eligible"},
        "min_duration": {
            "default": 1.0,
            "type": "float",
            "help": "videos shorter than this (seconds) are not eligible",
        },
        "decode_check": {
            "default": True,
            "type": "bool",
            "help": "decode every frame to find corrupt or cut-off files",
        },
        "timeout": {"default": 1800, "type": "int", "help": "seconds per ffmpeg call"},
    },
    "description": """
    Checks that each video is readable and eligible, then encodes it into a ladder of standard H.264/AAC MP4s.

    ### Features
    - `video_check` on every video: ffprobe summary (codec, size, rotation, fps, audio, duration) plus problems:
      unreadable, no video stream, too short, too small, decode errors, duration different from `source_duration`.
    - `renditions` on eligible videos: one MP4 per ladder rung up to the source size (yuv420p, +faststart, AAC stereo).
      The top rung is a lossless remux when the source is already H.264/AAC in MP4/MOV at that size.
    - Every rendition is probed back (codec, size, duration, audio); any mismatch sets `transcode_error` and no renditions.
    - A lower rung that is not smaller than the rung above it is dropped (it saves nothing).

    ### Notes
    - Requires `ffmpeg` and `ffprobe` on PATH.
    - Never raises: problems are reported in the media properties.
    """,
}
