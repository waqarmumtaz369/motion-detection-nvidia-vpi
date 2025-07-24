import sys
import vpi
import numpy as np
from argparse import ArgumentParser

import cv2
import time
import threading
from rabbitmq_client import RabbitMQClient
import json

# --- GStreamer display pipeline helper ---
class GstDisplay:
    def __init__(self, width, height, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.appsrc = None
        self._init_pipeline()

    def _init_pipeline(self):
        # Use autovideosink for portability, nveglglessink for Jetson
        pipeline_str = (
            f"appsrc name=src is-live=true block=true format=3 caps=video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1 "
            f"! videoconvert ! autovideosink sync=false"
        )
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsrc = self.pipeline.get_by_name('src')
        self.pipeline.set_state(Gst.State.PLAYING)

    def push(self, frame):
        # frame: numpy array (BGR, uint8)
        import ctypes
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        # Set timestamp and duration for smooth playback
        duration = int(1e9 / self.fps)
        buf.pts = buf.dts = int(time.time() * 1e9)
        buf.duration = duration
        retval = self.appsrc.emit('push-buffer', buf)
        if retval != Gst.FlowReturn.OK:
            print(f"[WARNING] GStreamer push-buffer returned {retval}")

    def close(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

import atexit

# GStreamer imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Motion detection parameters
# MOTION_PIXEL_THRESHOLD = 1000  # Minimum number of motion pixels to trigger detection (tune as needed)
CONTOUR_AREA_THRESHOLD = 100  # Minimum area for a contour to be considered motion (optional, for bounding boxes)

# Load YOLOv8 model 
YOLO_MODEL_PATH = "yolov8s.pt"

from ultralytics import YOLO
model = YOLO(YOLO_MODEL_PATH)


# ----------------------------
# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('backend', choices=['cpu','cuda'], help='Backend to be used for processing')
parser.add_argument('input', help='Input video to be denoised (file path or URI)')
parser.add_argument('--display', action='store_true', help='Enable video display (default: False)')
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


def build_nvidia_gst_pipeline(uri, target_w=None, target_h=None):
    """
    Build a GStreamer pipeline string using NVIDIA hardware decoder if possible.
    Adds videoscale and capsfilter to resize frames if needed, preserving aspect ratio.
    
    Args:
        uri: Input URI or file path
        target_w: Target width (if None, no resize)
        target_h: Target height (if None, no resize)
    """
    import os
    from urllib.parse import urlparse
    nvh264_available = Gst.ElementFactory.find("nvh264dec") is not None
    nvv4l2_available = Gst.ElementFactory.find("nvv4l2decoder") is not None
    parsed = urlparse(uri)
    
    # Build caps string with or without resize
    if target_w and target_h:
        resize_caps = f"videoscale ! video/x-raw,format=RGB,width={target_w},height={target_h} !"
    else:
        resize_caps = "videoconvert ! video/x-raw,format=RGB !"
    if parsed.scheme in ("file", ""):
        path = parsed.path if parsed.scheme else uri
        ext = os.path.splitext(path)[1].lower()
        if ext in [".mp4", ".mov", ".mkv", ".avi"]:
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
                    f"videoconvert ! {resize_caps} appsink name=sink"
                )
            elif nvv4l2_available:
                print("[INFO] Using NVIDIA nvv4l2decoder for hardware-accelerated decoding (Jetson/embedded).")
                pipeline = (
                    f"filesrc location=\"{path}\" ! {demux} ! h264parse ! nvv4l2decoder ! "
                    f"videoconvert ! {resize_caps} appsink name=sink"
                )
            else:
                print("[WARNING] No NVIDIA hardware decoder found. Falling back to software decoding (avdec_h264). To enable hardware acceleration, install NVIDIA GStreamer plugins.")
                pipeline = (
                    f"filesrc location=\"{path}\" ! {demux} ! h264parse ! avdec_h264 ! "
                    f"videoconvert ! {resize_caps} appsink name=sink"
                )
            return pipeline
        else:
            return f"uridecodebin uri={uri} ! videoconvert ! {resize_caps} appsink name=sink"
    else:
        return f"uridecodebin uri={uri} ! videoconvert ! {resize_caps} appsink name=sink"

def gstreamer_frame_generator_raw(uri):
    """
    Base GStreamer pipeline that returns frames at original size.
    """
    import datetime
    # Create a pipeline without resizing to get original frame size
    pipeline_str = build_nvidia_gst_pipeline(uri)
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name('sink')
    appsink.set_property('emit-signals', True)
    appsink.set_property('sync', False)
    pipeline.set_state(Gst.State.PLAYING)
    try:
        sample = appsink.emit('try-pull-sample', Gst.SECOND)
        if sample:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = structure.get_value('width')
            height = structure.get_value('height')
            success, mapinfo = buf.map(Gst.MapFlags.READ)
            if success:
                try:
                    data = mapinfo.data
                    arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                    yield arr
                finally:
                    buf.unmap(mapinfo)
    finally:
        pipeline.set_state(Gst.State.NULL)

def gstreamer_frame_generator(uri):
    """
    GStreamer pipeline to read frames and yield them as numpy arrays (BGR), using NVIDIA decoder if possible.
    Automatically resizes if width > 640 while preserving aspect ratio.
    """
    # First get the original size to calculate aspect ratio if needed
    orig_size = None
    for frame in gstreamer_frame_generator_raw(uri):
        orig_size = (frame.shape[1], frame.shape[0])  # width, height
        break
    if not orig_size:
        raise RuntimeError("Could not get original frame size")
    
    # Calculate target size maintaining aspect ratio
    orig_w, orig_h = orig_size
    if orig_w > 640:
        target_w = 640
        target_h = int(orig_h * (640 / orig_w))
        print(f"[INFO] Resizing from {orig_w}x{orig_h} to {target_w}x{target_h} to maintain aspect ratio")
    else:
        target_w, target_h = orig_w, orig_h
        print(f"[INFO] No resizing needed, original size {orig_w}x{orig_h} (width <= 640)")
    
    # Now create the pipeline with the correct size
    import datetime
    pipeline_str = build_nvidia_gst_pipeline(uri, target_w, target_h)
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
                # Convert RGB (from GStreamer) to BGR (for OpenCV compatibility)
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
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


class DisplayWrapper:
    """Wrapper class to handle display logic based on display flag"""
    def __init__(self, width, height, fps=30, enable_display=False):
        self.display_enabled = enable_display
        self.gst_display = None
        if self.display_enabled:
            self.gst_display = GstDisplay(width, height, fps)
            atexit.register(self.close)
    
    def push(self, frame):
        if self.display_enabled and self.gst_display:
            self.gst_display.push(frame)
    
    def close(self):
        if self.display_enabled and self.gst_display:
            self.gst_display.close()

input_uri = to_gst_uri(args.input)
inSize = get_video_size(input_uri)
if inSize == (0, 0):
    print(f"Error: Could not open input video file '{args.input}' via GStreamer")
    sys.exit(1)

# --- Initialize display handler ---
display = DisplayWrapper(inSize[0], inSize[1], fps=30, enable_display=args.display)

#--------------------------------------------------------------
# Create the Background Subtractor object using the backend specified by the user
with backend:
    bgsub = vpi.BackgroundSubtractor(inSize, vpi.Format.BGR8)

#--------------------------------------------------------------
# Main processing loop using GStreamer
start_time = time.time()
idxFrame = 0
# Initialize RabbitMQ client (before main loop)
rabbit_client = None
try:
    rabbit_client = RabbitMQClient()
    # Test the connection
    if rabbit_client.connect():
        print("[INFO] RabbitMQ client initialized and connected successfully")
    else:
        print("[ERROR] RabbitMQ client failed to connect")
        rabbit_client = None
except Exception as e:
    print(f"[ERROR] Could not initialize RabbitMQ client: {e}")
    rabbit_client = None

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
    # motion_pixels = cv2.countNonZero(motion_mask)

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
    # if motion_pixels > MOTION_PIXEL_THRESHOLD and len(motion_rois) > 0:
    if len(motion_rois) > 0:
        # Run YOLO once on the whole frame (RGB)
        frame_rgb = cv2.cvtColor(cvFrame, cv2.COLOR_BGR2RGB)
        results = model.predict(frame_rgb, imgsz=320, conf=0.25, verbose=False)
        detection_results = []  # To collect detection info for this frame
        object_names = []
        bb_coords = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.70:
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
                        # Collect detection info
                        object_names.append(label)
                        bb_coords.append([int(bx1), int(by1), int(bx2), int(by2)])
                        break
    # After all detections for this frame, build the JSON object
    detection_json = {
        "camera_number": 1,  # Default, or parse from input if needed
        "frame_number": idxFrame,
        "alert_type": "motion_detection",
        "no_of_objects": len(object_names),
        "list_of_object_names": object_names,
        "bb_coord_of_objects": bb_coords
    }
    # Append detection_json to results.json (newline-delimited JSON)
    with open("results.json", "a") as f:
        f.write(json.dumps(detection_json) + "\n")
    # Send to RabbitMQ in a separate thread (non-blocking)
    def send_to_rabbitmq(data):
        if rabbit_client:
            try:
                success = rabbit_client.publish_frame_details(data)
                if success:
                    print(f"[INFO] Frame {idxFrame} data sent to RabbitMQ successfully")
                else:
                    print(f"[WARNING] Failed to send frame {idxFrame} data to RabbitMQ")
            except Exception as e:
                print(f"[ERROR] Exception while sending frame {idxFrame} to RabbitMQ: {e}")
        else:
            print(f"[WARNING] RabbitMQ client not available, skipping frame {idxFrame}")
    
    # Only send to RabbitMQ if there are detected objects
    if len(object_names) > 0:
        threading.Thread(target=send_to_rabbitmq, args=(detection_json,), daemon=True).start()
    # --- END OBJECT DETECTION LOGIC ---


    # Display the processed frame if display is enabled
    display.push(cvFrame)
    
    # Handle keyboard input if display is enabled
    if args.display and cv2.waitKey(1) & 0xFF == ord('q'):
        break


end_time = time.time()
if args.display:
    cv2.destroyAllWindows()
display.close()

# Close RabbitMQ connection
if rabbit_client:
    rabbit_client.close()
    print("[INFO] RabbitMQ connection closed")

total_time = end_time - start_time
print("Total time taken to process the video: {:.2f} seconds".format(total_time))
print("Total number of frames processed: {}".format(idxFrame))
if total_time > 0:
    fps_processed = idxFrame / total_time
    print("Processing FPS: {:.2f}".format(fps_processed))
else:
    print("Processing FPS: N/A (zero time elapsed)")
