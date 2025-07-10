# Motion Detection with NVIDIA VPI and GStreamer Hardware Acceleration

This project demonstrates real-time motion detection in video streams using the Background Subtractor from NVIDIA Vision Programming Interface (VPI). See the [VPI Background Subtractor documentation](https://docs.nvidia.com/vpi/algo_background_subtractor.html) for more details. It leverages GStreamer for efficient video decoding, utilizing NVIDIA hardware acceleration when available.

## Features
- **Motion Detection**: Uses VPI's Background Subtractor to detect moving objects in video frames.
- **Hardware-Accelerated Decoding**: GStreamer pipelines are configured to use NVIDIA hardware decoders (e.g., `nvh264dec`, `nvv4l2decoder`) for fast frame extraction when available, with fallback to software decoding.
- **Bounding Boxes**: Draws bounding boxes around detected moving objects.
- **Performance Metrics**: Prints processing time and frames-per-second (FPS) statistics.

## Requirements
- Python 3.x
- NVIDIA VPI 3.x and Python bindings ([installation instructions](https://docs.nvidia.com/vpi/installation.html))
- OpenCV (`cv2`) with GStreamer support
- GStreamer (with Python GObject bindings)
- NumPy

### NVIDIA VPI Installation (Ubuntu)
Follow the [official NVIDIA VPI installation guide](https://docs.nvidia.com/vpi/installation.html). Below are summarized steps for Ubuntu 20.04/22.04 (x86_64):

1. Install the public key for the VPI repository:
   ```bash
   sudo apt install gnupg
   sudo apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc
   ```
2. Install packages needed to add a new apt repository:
   ```bash
   sudo apt install software-properties-common
   ```
3. Add the public repository server to the apt configuration:
   - For Ubuntu 20.04:
     ```bash
     sudo add-apt-repository 'deb https://repo.download.nvidia.com/jetson/x86_64/focal r36.4 main'
     ```
   - For Ubuntu 22.04:
     ```bash
     sudo add-apt-repository 'deb https://repo.download.nvidia.com/jetson/x86_64/jammy r36.4 main'
     ```
4. Update the local repository package list:
   ```bash
   sudo apt update
   ```
5. Install the VPI package and its dependencies:
   ```bash
   sudo apt install libnvvpi3 vpi3-dev vpi3-samples
   ```
6. For Python bindings:
   - Python 3.9 (Ubuntu 20.04/22.04):
     ```bash
     sudo apt install python3.9-vpi3
     ```
   - Python 3.10 (Ubuntu 22.04 only):
     ```bash
     sudo apt install python3.10-vpi3
     ```
   - In both cases, install numpy (if not already):
     ```bash
     pip install numpy
     ```
7. (Optional) For cross-compilation targeting aarch64-l4t:
   ```bash
   sudo apt install vpi3-cross-aarch64-l4t
   ```

### GStreamer Installation
Install GStreamer and development plugins:
```bash
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl \
    gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
```

### OpenCV Installation
Install OpenCV with GStreamer support:
```bash
sudo apt-get install python3-opencv
```
> **Note:** For more robust installation or if you need the latest features, consider building OpenCV from source with GStreamer support enabled.

### Additional Python Dependencies
```bash
pip install "numpy<2.0"
```

## Usage
Run the script with the desired backend and input video:

```
python3 main.py <backend> <input_video>
```
- `<backend>`: `cpu` or `cuda` (for GPU acceleration)
- `<input_video>`: Path to the input video file (e.g., `videos/sample001.mp4`)

Example:
```
python3 main.py cuda videos/sample001.mp4
```

## How It Works
- The script builds a GStreamer pipeline to decode video frames, preferring NVIDIA hardware decoders if available.
- Each frame is processed by VPI's Background Subtractor to generate a foreground mask.
- Motion is detected by thresholding the mask and counting motion pixels.
- Bounding boxes are drawn around detected moving regions.
- The processed video is displayed in real time. Press `q` to exit early.

## Notes
- For best performance, ensure NVIDIA drivers and GStreamer plugins are properly installed.
- The motion detection thresholds can be tuned in `main.py` for different scenarios.
- The project is licensed under the Apache License 2.0 (see `LICENSE`).

## Directory Structure
```
LICENSE           # License file (Apache 2.0)
main.py           # Main script for motion detection
README.md         # This file
videos/           # Sample video files
```

## References
- [NVIDIA VPI Documentation](https://docs.nvidia.com/vpi/)
- [VPI Background Subtractor](https://docs.nvidia.com/vpi/algo_background_subtractor.html)
- [NVIDIA VPI Installation Guide](https://docs.nvidia.com/vpi/installation.html)
- [GStreamer Documentation](https://gstreamer.freedesktop.org/documentation/)
