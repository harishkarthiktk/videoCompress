"""Constants for video processing configuration"""

# Video bitrate thresholds (kbps)
BITRATE_THRESHOLDS = {
    'HIGH': 8000,       # >8 Mbps: aggressive compression
    'MEDIUM': 2000,     # 2-8 Mbps: moderate compression
    'LOW': 1000,        # <1 Mbps: conservative
}

# Resolution thresholds (pixels)
RESOLUTION_THRESHOLDS = {
    '4K': 2160,
    'HD': 1080,
    'SD': 720,
}

# Audio bitrate settings (kbps)
AUDIO_BITRATES = {
    'MP3_HIGH': 320,
    'AAC_HIGH': 256,
    'AAC_DEFAULT': 128,
    'MIN': 64,
}

# Quality settings for compression
QUALITY = {
    'HIGH_CRF': 23,
    'MEDIUM_CRF': 25,
    'LOW_CRF': 28,
    'MAX_BITRATE_LIMIT': 10000,    # kbps (10 Mbps)
    'MIN_AUDIO_BITRATE': 64,        # kbps
}

# Processing timeouts (seconds)
TIMEOUTS = {
    'GPU_DETECTION': 10,
    'FFPROBE': 30,
    'FFMPEG_CONVERSION': 3600,  # 1 hour
}

# Supported file formats
SUPPORTED_VIDEO_FORMATS = (
    '.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm', '.m4v',
    '.mpg', '.3gp', '.MP4', '.MKV', '.AVI', '.MOV', '.FLV', '.WMV',
    '.WEBM', '.M4V', '.MPG', '.3GP'
)

SUPPORTED_AUDIO_FORMATS = ('mp3', 'aac', 'flac')

# Codec optimization
INEFFICIENT_CODECS = {'mpeg4', 'wmv2', 'msmpeg4'}
LONG_DURATION_THRESHOLD = 3600  # seconds
MAX_DOWNSCALE_PERCENT = 0.5     # Maximum resolution reduction (50%)
