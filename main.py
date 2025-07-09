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
# Open input and output videos

inVideo = cv2.VideoCapture(args.input)

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
inSize = (int(inVideo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(inVideo.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = inVideo.get(cv2.CAP_PROP_FPS)

outVideoFGMask = cv2.VideoWriter('fgmask_python'+str(sys.version_info[0])+'_'+args.backend+'.mp4',
                                 fourcc, fps, inSize)

outVideoBGImage = cv2.VideoWriter('bgimage_python'+str(sys.version_info[0])+'_'+args.backend+'.mp4',
                                  fourcc, fps, inSize)

#--------------------------------------------------------------
# Create the Background Subtractor object using the backend specified by the user
with backend:
    bgsub = vpi.BackgroundSubtractor(inSize, vpi.Format.BGR8)

#--------------------------------------------------------------
# Main processing loop
start_time = time.time()
idxFrame = 0
while True:
    print("Processing frame {}".format(idxFrame))
    idxFrame+=1

    # Read one input frame
    ret, cvFrame = inVideo.read()
    if not ret:
        break

    # Get the foreground mask and background image estimates
    fgmask, bgimage = bgsub(vpi.asimage(cvFrame, vpi.Format.BGR8), learnrate=0.01)

    # Mask needs to be converted to BGR8 for output
    fgmask = fgmask.convert(vpi.Format.BGR8, backend=vpi.Backend.CUDA);

    cv2.imshow('Foreground Mask', fgmask.cpu())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Write images to output videos
    # with fgmask.rlock_cpu(), bgimage.rlock_cpu():
    #     outVideoFGMask.write(fgmask.cpu())
    #     outVideoBGImage.write(bgimage.cpu())

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