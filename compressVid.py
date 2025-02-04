import os, sys, subprocess, platform
import concurrent.futures
from tqdm import tqdm

# supported video formats
SUPPORTED_FORMATS = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".MP4", ".MKV", ".AVI", ".MOV", ".FLV", ".WMV", ".WEBM", ".M4V")

# target resolution & bitrate
TARGET_RESOLUTION = (1920, 1080)
TARGET_BITRATE = 5000  # in kbps

'''
The script converts {{SUPPORTED FORMATS}} to .mp4 format with optimal compression.
It uses GPU acceleration if available and skips a file if _conv.mp4 already exists.
'''

# detect gpu type (nvidia > intel > amd > apple > cpu)
def detect_gpu():
    system = platform.system()
    gpu_type = "CPU"

    try:
        if system == "Windows":
            output = subprocess.check_output("wmic path win32_VideoController get Name", shell=True, text=True)
        elif system == "Linux":
            output = subprocess.check_output("lspci | grep -i vga", shell=True, text=True)
        elif system == "Darwin":  # macos
            output = subprocess.check_output("system_profiler SPDisplaysDataType", shell=True, text=True)
        else:
            return gpu_type  # default to cpu if os is unknown

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

# get video resolution & bitrate
def get_video_info(file):
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,bit_rate", "-of", "default=noprint_wrappers=1", file
        ]
        output = subprocess.check_output(cmd, text=True).split("\n")

        width, height, bitrate = None, None, None
        for line in output:
            if "width=" in line:
                width = int(line.split("=")[1])
            if "height=" in line:
                height = int(line.split("=")[1])
            if "bit_rate=" in line:
                bitrate = int(line.split("=")[1]) // 1000  # convert bps to kbps

        return width, height, bitrate

    except Exception as e:
        print(f"Failed to get video info for {file}: {e}")
        return None, None, None

# convert video with detected gpu acceleration
def convert_video(file, gpu_type, progress_bar):
    filename, ext = os.path.splitext(file)
    output_file = f"{filename}_conv.mp4"

    # skip if already converted
    if os.path.exists(output_file):
        progress_bar.update(1)
        print(f"Skipping {file}, already converted.")
        return

    # get input resolution & bitrate
    width, height, bitrate = get_video_info(file)

    # skip if input resolution & bitrate are already lower than target
    if width is not None and height is not None and bitrate is not None:
        if (width, height) <= TARGET_RESOLUTION and bitrate <= TARGET_BITRATE:
            progress_bar.update(1)
            print(f"Skipping {file}, already optimal (Resolution: {width}x{height}, Bitrate: {bitrate} kbps).")
            return

    print(f"Processing: {file} -> {output_file} using {gpu_type}")

    # maintain aspect ratio while scaling
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", file, "-vf",
        "scale='if(gt(iw,1920),1920,if(lt(iw,1280),1280,iw))':'if(gt(ih,1080),1080,if(lt(ih,720),720,ih))':force_original_aspect_ratio=decrease"
    ]

    if gpu_type == "NVIDIA":
        ffmpeg_cmd += ["-c:v", "h264_nvenc", "-preset", "slow", "-cq", "23"]
    elif gpu_type == "INTEL":
        ffmpeg_cmd += ["-c:v", "h264_qsv", "-preset", "slow", "-b:v", "5000k"]
    elif gpu_type == "AMD":
        ffmpeg_cmd += ["-c:v", "h264_amf", "-quality", "slow", "-b:v", "5000k"]
    elif gpu_type == "APPLE":  # macos apple silicon (m1, m2)
        ffmpeg_cmd += ["-c:v", "h264_videotoolbox", "-b:v", "5000k"]
    else:
        ffmpeg_cmd += ["-c:v", "libx264", "-crf", "23", "-preset", "medium"]

    ffmpeg_cmd += ["-c:a", "aac", "-b:a", "128k", "-progress", "pipe:1", "-stats", output_file]

    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # read ffmpeg output line by line
    for line in process.stderr:
        if "frame=" in line:
            progress_bar.set_description(f"Processing {file}")

    process.wait()
    progress_bar.update(1)

# run conversions in parallel
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

    with tqdm(total=len(video_files), desc="Overall Progress") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(lambda file: convert_video(file, gpu_type, progress_bar), video_files)

    print("All files have been processed!")

if __name__ == "__main__":
    main()
