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

# Motion detection parameters
MOTION_PIXEL_THRESHOLD = 500  # Minimum number of motion pixels to trigger detection (tune as needed)
CONTOUR_AREA_THRESHOLD = 100  # Minimum area for a contour to be considered motion (optional, for bounding boxes)

# ----------------------------
# Parse command line arguments

parser = ArgumentParser()
parser.add_argument('backend', choices=['cpu','cuda'],
                    help='Backend to be used for processing')

parser.add_argument('input',
                    help='Input video to be denoised')

args = parser.parse_args();

if args.backend == 'cuda':
    backend = vpi.Backend.CUDA
else:
    assert args.backend == 'cpu'
    backend = vpi.Backend.CPU

# -----------------------------
# Open input video and check
inVideo = cv2.VideoCapture(args.input)
if not inVideo.isOpened():
    print(f"Error: Could not open input video file '{args.input}'")
    sys.exit(1)
inSize = (int(inVideo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(inVideo.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = inVideo.get(cv2.CAP_PROP_FPS)

#--------------------------------------------------------------
# Create the Background Subtractor object using the backend specified by the user
with backend:
    bgsub = vpi.BackgroundSubtractor(inSize, vpi.Format.BGR8)

#--------------------------------------------------------------
# Main processing loop
start_time = time.time()
idxFrame = 0
while True:
    # print("Processing frame {}".format(idxFrame))
    idxFrame+=1

    # Read one input frame
    ret, cvFrame = inVideo.read()
    if not ret:
        idxFrame -= 1  # Don't count the last increment if no frame was read
        break

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

    if motion_pixels > MOTION_PIXEL_THRESHOLD:
        print(f"Motion detected in frame {idxFrame} (pixels: {motion_pixels})")
    else:
        print(f"No significant motion in frame {idxFrame}")

    # (Optional) Draw bounding boxes around moving objects
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > CONTOUR_AREA_THRESHOLD:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(cvFrame, (x, y), (x+w, y+h), (0,255,0), 2)
    # --- END MOTION DETECTION LOGIC ---

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
