# videoCompress
- A simple commandline utility using python to compress video files to a smaller file, while keeping optimized quality.

## Requirement:
- Videos shot on the phone are large, and often the memory is more important than the absolute clarity of the video.
- All utils available online focus on UI and variable options, but I only care for a simple utility.
- Either an argument can passed to operate on a different folder, or the script will operate in the location where the script is present.

## Future Enhancements:
- Add a GUI using Tkinter for path selection alone, since it adds ease.
- Multiprocessing is implemented using multithread and concurrent futures, it could be optimized.
- The TQDM shows overall progress, would be nice to show individual process as well.
- Tested only on Mac-M1 chip, and Windows with Nvidia card for hardware acceleration, should test on other platforms.