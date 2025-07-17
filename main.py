# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import sys
import vpi
import numpy as np
from argparse import ArgumentParser
import cv2
import time

# GStreamer imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Motion detection parameters
MOTION_PIXEL_THRESHOLD = 1000  # Minimum number of motion pixels to trigger detection (tune as needed)
CONTOUR_AREA_THRESHOLD = 100  # Minimum area for a contour to be considered motion (optional, for bounding boxes)

# Load YOLOv8 model (nano version)
# YOLO_MODEL_PATH = "yolov8s.pt"
YOLO_MODEL_PATH = "yolov8n.pt"
from ultralytics import YOLO
model = YOLO(YOLO_MODEL_PATH)


# ----------------------------
# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('backend', choices=['cpu','cuda'], help='Backend to be used for processing')
parser.add_argument('input', help='Input video to be denoised (file path or URI)')
args = parser.parse_args()

if args.backend == 'cuda':
    backend = vpi.Backend.CUDA
else:
    assert args.backend == 'cpu'
    backend = vpi.Backend.CPU


# -----------------------------
# GStreamer initialization with timing
import datetime
gst_init_start = time.time()
print(f"[{datetime.datetime.now().isoformat()}] Starting GStreamer initialization...")
Gst.init(None)
gst_init_end = time.time()
print(f"[{datetime.datetime.now().isoformat()}] GStreamer initialized. Took {gst_init_end-gst_init_start:.3f} seconds.")

def build_nvidia_gst_pipeline(uri):
    """
    Build a GStreamer pipeline string using NVIDIA hardware decoder if possible.
    Only supports local files (mp4, mkv, mov, avi) with H.264/H.265 for now.
    Falls back to software decoding if nvv4l2decoder is not available.
    """
    import os
    from urllib.parse import urlparse
    # Prefer nvh264dec (x86_64 RTX) over nvv4l2decoder (Jetson/embedded)
    nvh264_available = Gst.ElementFactory.find("nvh264dec") is not None
    nvv4l2_available = Gst.ElementFactory.find("nvv4l2decoder") is not None
    # Determine if local file or URI
    parsed = urlparse(uri)
    if parsed.scheme in ("file", ""):
        # Local file
        path = parsed.path if parsed.scheme else uri
        ext = os.path.splitext(path)[1].lower()
        if ext in [".mp4", ".mov", ".mkv", ".avi"]:
            # Use qtdemux for mp4/mov, matroskademux for mkv, avidemux for avi
            if ext in [".mp4", ".mov"]:
                demux = "qtdemux"
            elif ext == ".mkv":
                demux = "matroskademux"
            elif ext == ".avi":
                demux = "avidemux"
            else:
                demux = "qtdemux"
            if nvh264_available:
                print("[INFO] Using NVIDIA nvh264dec for hardware-accelerated decoding (x86_64, RTX).")
                pipeline = (
                    f"filesrc location=\"{path}\" ! {demux} ! h264parse ! nvh264dec ! "
                    "videoconvert ! video/x-raw,format=BGR ! appsink name=sink"
                )
            elif nvv4l2_available:
                print("[INFO] Using NVIDIA nvv4l2decoder for hardware-accelerated decoding (Jetson/embedded).")
                pipeline = (
                    f"filesrc location=\"{path}\" ! {demux} ! h264parse ! nvv4l2decoder ! "
                    "videoconvert ! video/x-raw,format=BGR ! appsink name=sink"
                )
            else:
                print("[WARNING] No NVIDIA hardware decoder found. Falling back to software decoding (avdec_h264). To enable hardware acceleration, install NVIDIA GStreamer plugins.")
                pipeline = (
                    f"filesrc location=\"{path}\" ! {demux} ! h264parse ! avdec_h264 ! "
                    "videoconvert ! video/x-raw,format=BGR ! appsink name=sink"
                )
            return pipeline
        else:
            # Fallback to uridecodebin for other formats
            return f"uridecodebin uri={uri} ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink"
    else:
        # For network streams, fallback to uridecodebin
        return f"uridecodebin uri={uri} ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink"

def gstreamer_frame_generator(uri):
    """
    GStreamer pipeline to read frames and yield them as numpy arrays (BGR), using NVIDIA decoder if possible.
    """
    import datetime
    pipeline_str = build_nvidia_gst_pipeline(uri)
    pipeline_start = time.time()
    print(f"[{datetime.datetime.now().isoformat()}] Starting GStreamer pipeline creation and set_state...")
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name('sink')
    appsink.set_property('emit-signals', True)
    appsink.set_property('sync', False)
    pipeline.set_state(Gst.State.PLAYING)
    pipeline_end = time.time()
    print(f"[{datetime.datetime.now().isoformat()}] GStreamer pipeline ready. Took {pipeline_end-pipeline_start:.3f} seconds.")
    try:
        while True:
            sample = appsink.emit('try-pull-sample', Gst.SECOND)
            if sample is None:
                break
            buf = sample.get_buffer()
            caps = sample.get_caps()
            arr = None
            # Extract width, height
            structure = caps.get_structure(0)
            width = structure.get_value('width')
            height = structure.get_value('height')
            # Get data
            success, mapinfo = buf.map(Gst.MapFlags.READ)
            if not success:
                break
            try:
                data = mapinfo.data
                arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            finally:
                buf.unmap(mapinfo)
            yield arr
    finally:
        pipeline.set_state(Gst.State.NULL)

# Helper to get video size from first frame
def get_video_size(uri):
    for frame in gstreamer_frame_generator(uri):
        return (frame.shape[1], frame.shape[0])
    return (0, 0)

# Accept both file paths and URIs
def to_gst_uri(path):
    if path.startswith('http://') or path.startswith('https://') or path.startswith('file://'):
        return path
    import pathlib
    return pathlib.Path(path).absolute().as_uri()

input_uri = to_gst_uri(args.input)
inSize = get_video_size(input_uri)
if inSize == (0, 0):
    print(f"Error: Could not open input video file '{args.input}' via GStreamer")
    sys.exit(1)

#--------------------------------------------------------------
# Create the Background Subtractor object using the backend specified by the user
with backend:
    bgsub = vpi.BackgroundSubtractor(inSize, vpi.Format.BGR8)

#--------------------------------------------------------------
# Main processing loop using GStreamer
start_time = time.time()
idxFrame = 0
for cvFrame in gstreamer_frame_generator(input_uri):
    idxFrame += 1
    # Make frame writable for OpenCV drawing
    cvFrame = cvFrame.copy()
    # Get the foreground mask and background image estimates
    fgmask, bgimage = bgsub(vpi.asimage(cvFrame, vpi.Format.BGR8), learnrate=0.01)

    # Mask needs to be converted to BGR8 for output
    fgmask_bgr = fgmask.convert(vpi.Format.BGR8, backend=vpi.Backend.CUDA)

    # --- MOTION DETECTION LOGIC ---
    # Convert fgmask to grayscale for analysis
    fgmask_gray = cv2.cvtColor(fgmask_bgr.cpu(), cv2.COLOR_BGR2GRAY)

    # Apply threshold to get binary mask
    _, motion_mask = cv2.threshold(fgmask_gray, 127, 255, cv2.THRESH_BINARY)

    # Count non-zero (white) pixels
    motion_pixels = cv2.countNonZero(motion_mask)

    # (Optional) Draw bounding boxes around moving objects
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_rois = []
    for cnt in contours:
        if cv2.contourArea(cnt) > CONTOUR_AREA_THRESHOLD:
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(cvFrame, (x, y), (x+w, y+h), (0,255,0), 2)
            motion_rois.append((x, y, w, h))
    # --- END MOTION DETECTION LOGIC ---

    # --- OBJECT DETECTION LOGIC ---
    # Only run YOLOv8 if motion threshold is exceeded
    if motion_pixels > MOTION_PIXEL_THRESHOLD and len(motion_rois) > 0:
        # Run YOLO once on the whole frame (RGB)
        frame_rgb = cv2.cvtColor(cvFrame, cv2.COLOR_BGR2RGB)
        results = model.predict(frame_rgb, imgsz=320, conf=0.25, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.79:
                    continue
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                bx1, by1, bx2, by2 = xyxy
                # Check if the center of the detection is inside any motion ROI
                cx = int((bx1 + bx2) / 2)
                cy = int((by1 + by2) / 2)
                for (mx, my, mw, mh) in motion_rois:
                    if mx <= cx <= mx+mw and my <= cy <= my+mh:
                        label = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)
                        cv2.rectangle(cvFrame, (bx1, by1), (bx2, by2), (0,0,255), 2)
                        cv2.putText(cvFrame, f"{label} {conf:.2f}", (bx1, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        break
    # --- END OBJECT DETECTION LOGIC ---

    # Display the processed frame with bounding boxes (cvFrame)
    cv2.imshow('Motion Detection', cvFrame)
    # Optionally, display the mask as well:
    # cv2.imshow('Foreground Mask', motion_mask)
    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
cv2.destroyAllWindows()
total_time = end_time - start_time
print("Total time taken to process the video: {:.2f} seconds".format(total_time))
print("Total number of frames processed: {}".format(idxFrame))
if total_time > 0:
    fps_processed = idxFrame / total_time
    print("Processing FPS: {:.2f}".format(fps_processed))
else:
    print("Processing FPS: N/A (zero time elapsed)")
