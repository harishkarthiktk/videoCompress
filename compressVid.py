import os, sys, subprocess, platform, json

# Supported video formats
SUPPORTED_FORMATS = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v", "mpg", "3gp", ".MP4", ".MKV", ".AVI", ".MOV", ".FLV", ".WMV", ".WEBM", ".M4V", "MPG", "3GP")

# Target bitrate limits
MAX_BITRATE = 10000  # Max total bitrate in kbps (10 Mbps)
COMPRESSION_FACTOR = 0.8  # Reduce bitrate by 20%
MAX_DOWNSCALE_PERCENT = 0.2  # 20% max resolution reduction

'''
This script converts videos to .mp4 format with optimal compression.
- Uses GPU acceleration if available.
- Skips processing if _conv.mp4 already exists.
- Ensures at least 20% compression if bitrate ≤ 10 Mbps.
'''

# Detect GPU type (NVIDIA > Intel > AMD > Apple > CPU)
def detect_gpu():
    system = platform.system()
    gpu_type = "CPU"

    try:
        if system == "Windows":
            output = subprocess.check_output("wmic path win32_VideoController get Name", shell=True, text=True)
        elif system == "Linux":
            output = subprocess.check_output("lspci | grep -i vga", shell=True, text=True)
        elif system == "Darwin":  # macOS
            output = subprocess.check_output("system_profiler SPDisplaysDataType", shell=True, text=True)
        else:
            return gpu_type  # Default to CPU if OS is unknown

        output = output.lower()

        if "nvidia" in output:
            gpu_type = "NVIDIA"
        elif "intel" in output:
            gpu_type = "INTEL"
        elif "amd" in output or "radeon" in output:
            gpu_type = "AMD"
        elif "apple" in output or "m1" in output or "m2" in output:
            gpu_type = "APPLE"
    except Exception as e:
        print(f"GPU Detection Failed: {e}")

    return gpu_type

# Get video resolution & bitrate
def get_video_info(file):
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,bit_rate",
            "-of", "json",
            file
        ]
        output = subprocess.check_output(cmd, text=True)
        data = json.loads(output)
        stream = data.get('streams', [{}])[0]

        width = stream.get('width')
        height = stream.get('height')
        bitrate = stream.get('bit_rate')

        # Convert bitrate to kbps if present
        if bitrate is not None:
            bitrate = int(bitrate) // 1000

        return width, height, bitrate

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None


# Convert video with detected GPU acceleration
def convert_video(file, gpu_type):
    filename, ext = os.path.splitext(file)
    output_file = f"{filename}_conv.mp4"

    # Skip if already converted
    if os.path.exists(output_file):
        print(f"Skipping {file}, already converted.")
        return

    # Get input resolution & bitrate
    width, height, bitrate = get_video_info(file)
    print(width, height, bitrate)
    if bitrate is None:
        bitrate = MAX_BITRATE

    # Determine new bitrate
    if bitrate:
        if bitrate > MAX_BITRATE:
            target_bitrate = MAX_BITRATE  # Limit to 10 Mbps
        else:
            target_bitrate = int(bitrate * COMPRESSION_FACTOR)  # Reduce by 20%

    # Calculate the new resolution (maximum 20% downscale)
    new_width = int(width * (1 - MAX_DOWNSCALE_PERCENT))
    new_height = int(height * (1 - MAX_DOWNSCALE_PERCENT))

    print(f"Processing: {file} -> {output_file} using {gpu_type}")

    # Downscale while maintaining aspect ratio and limit bitrate
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", file, "-vf",
        f"scale='min(iw,{new_width})':'min(ih,{new_height})':force_original_aspect_ratio=decrease"
    ]

    # Select GPU encoder if available
    if gpu_type == "NVIDIA":
        ffmpeg_cmd += ["-c:v", "h264_nvenc", "-preset", "slow", "-b:v", f"{target_bitrate}k"]
    elif gpu_type == "INTEL":
        ffmpeg_cmd += ["-c:v", "h264_qsv", "-preset", "slow", "-b:v", f"{target_bitrate}k"]
    elif gpu_type == "AMD":
        ffmpeg_cmd += ["-c:v", "h264_amf", "-quality", "slow", "-b:v", f"{target_bitrate}k"]
    elif gpu_type == "APPLE":  # macOS Apple Silicon (M1, M2)
        ffmpeg_cmd += ["-c:v", "h264_videotoolbox", "-b:v", f"{target_bitrate}k"]
    else:
        ffmpeg_cmd += ["-c:v", "libx264", "-crf", "23", "-preset", "medium", "-b:v", f"{target_bitrate}k"]

    # Set audio codec and final output
    ffmpeg_cmd += ["-c:a", "aac", "-b:a", "128k", "-stats", output_file]

    subprocess.run(ffmpeg_cmd)  # Run FFmpeg with live progress output

# Run conversions sequentially
def main():
    gpu_type = detect_gpu()

    arg_path = sys.argv[1] if len(sys.argv) > 1 else None
    if arg_path:
        print(f"Running script in path: {arg_path}")
        os.chdir(arg_path)
    else:
        print("No path was provided, running from script's root.")

    video_files = [f for f in os.listdir() if f.endswith(SUPPORTED_FORMATS)]
    if not video_files:
        print("No video files found.")
        return

    # Sequential processing of video files
    for file in video_files:
        convert_video(file, gpu_type)

    print("All files have been processed!")

if __name__ == "__main__":
    main()
